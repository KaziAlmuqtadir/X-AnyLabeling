import os
import cv2
import numpy as np

from PyQt6 import QtCore
from PyQt6.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .model import Model
from .types import AutoLabelingResult
from .engines.build_onnx_engine import OnnxBaseModel

# Default anchors from the upstream Yolo-FastestV2 COCO config
# (Yolo-FastestV2/data/coco.data). Only valid for models trained with
# that config; custom-trained models must set `anchors` explicitly.
_DEFAULT_ANCHORS = [
    12.64, 19.39, 37.88, 51.48, 55.71, 138.31,
    126.91, 78.23, 131.57, 214.55, 279.92, 258.87,
]  # fmt: skip

# Number of detection heads (scales) the Yolo-FastestV2 FPN outputs.
_NUM_SCALES = 2


class YoloFastestV2(Model):
    """Object detection model using Yolo-FastestV2

    Expects an ONNX model exported with the upstream repo's
    `pytorch2onnx.py`, which returns 2 output tensors (one per FPN scale),
    each of shape (1, H, W, 4*anchor_num + anchor_num + num_classes) with
    sigmoid/softmax already applied.
    """

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
            "classes",
            "iou_threshold",
            "conf_threshold",
        ]
        widgets = [
            "button_run",
            "input_conf",
            "edit_conf",
            "input_iou",
            "edit_iou",
            "toggle_preserve_existing_annotations",
            "button_classes_filter",
        ]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
        super().__init__(model_config, on_message)

        model_abs_path = self.get_model_abs_path(self.config, "model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    "Could not download or initialize Yolo-FastestV2 model.",
                )
            )

        self.net = OnnxBaseModel(model_abs_path, __preferred_device__)
        input_shape = self.net.get_input_shape()
        _, _, self.input_height, self.input_width = input_shape
        if not isinstance(self.input_height, int) or not isinstance(
            self.input_width, int
        ):
            raise ValueError(
                "Yolo-FastestV2: could not determine a static input size "
                f"from the ONNX model (got shape {input_shape}). Re-export "
                "the model without dynamic input axes."
            )

        self.classes = self.config["classes"]
        self.filter_classes = self.config.get("filter_classes", [])
        self.anchor_num = int(self.config.get("anchor_num", 3))

        anchors_cfg = self.config.get("anchors")
        if anchors_cfg is None:
            logger.warning(
                "yolo_fastestv2: no 'anchors' set in the model config; "
                "falling back to the default Yolo-FastestV2 COCO anchors. "
                "If this model was trained with custom anchors (via "
                "genanchors.py), detections will be inaccurate until you "
                "set 'anchors' explicitly in the config file."
            )
            anchors_cfg = _DEFAULT_ANCHORS
        anchors_flat = np.asarray(anchors_cfg, dtype=np.float32).reshape(-1)
        expected_size = _NUM_SCALES * self.anchor_num * 2
        if anchors_flat.size != expected_size:
            raise ValueError(
                "yolo_fastestv2: 'anchors' must contain "
                f"{expected_size} numbers ({_NUM_SCALES} scales x "
                f"{self.anchor_num} anchors x 2), got {anchors_flat.size}."
            )
        self.anchors = anchors_flat.reshape(_NUM_SCALES, self.anchor_num, 2)

        self.nms_thres = self.config["iou_threshold"]
        self.conf_thres = self.config["conf_threshold"]
        self.replace = True

    def set_auto_labeling_conf(self, value):
        """set auto labeling confidence threshold"""
        if value > 0:
            self.conf_thres = value

    def set_auto_labeling_iou(self, value):
        """set auto labeling iou threshold"""
        if value > 0:
            self.nms_thres = value

    def set_auto_labeling_preserve_existing_annotations_state(self, state):
        """Toggle the preservation of existing annotations based on the checkbox state."""
        self.replace = not state

    def set_auto_labeling_filter_classes(self, class_names):
        """Set filter classes by name."""
        if not class_names or len(class_names) == len(self.classes):
            self.filter_classes = []
        else:
            self.filter_classes = class_names

    def preprocess(self, input_image):
        """Yolo-FastestV2 is trained on a plain resize (no letterbox pad)."""
        src_h, src_w = input_image.shape[:2]
        resized = cv2.resize(
            input_image,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        blob = resized.transpose(2, 0, 1)[np.newaxis, :, :, :]
        blob = np.ascontiguousarray(blob).astype(np.float32) / 255.0
        ratio_w = src_w / self.input_width
        ratio_h = src_h / self.input_height
        return blob, ratio_w, ratio_h

    @staticmethod
    def _make_grid(h, w):
        grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        return grid_x.astype(np.float32), grid_y.astype(np.float32)

    def _decode_scale(self, out, scale_idx):
        """Decode one FPN scale's output into (boxes[cx,cy,w,h], conf, cls_id)."""
        na = self.anchor_num
        nc = len(self.classes)
        expected_depth = 4 * na + na + nc
        out = out[0]  # drop batch dim -> (H, W, depth)
        h, w, depth = out.shape
        if depth != expected_depth:
            raise ValueError(
                "yolo_fastestv2: unexpected output channel size "
                f"{depth} for scale {scale_idx} (expected {expected_depth} "
                f"= 4*anchor_num + anchor_num + num_classes with "
                f"anchor_num={na}, num_classes={nc}). Check that "
                "'classes'/'anchor_num' in the config match the model."
            )

        reg = out[..., : 4 * na].reshape(h, w, na, 4)
        obj = out[..., 4 * na : 4 * na + na].reshape(h, w, na)
        cls = out[..., 4 * na + na :]  # (h, w, nc), shared across anchors

        grid_x, grid_y = self._make_grid(h, w)
        stride = self.input_height / h

        cx = (reg[..., 0] * 2.0 - 0.5 + grid_x[:, :, None]) * stride
        cy = (reg[..., 1] * 2.0 - 0.5 + grid_y[:, :, None]) * stride
        anchor_w = self.anchors[scale_idx, :, 0].reshape(1, 1, na)
        anchor_h = self.anchors[scale_idx, :, 1].reshape(1, 1, na)
        bw = (reg[..., 2] * 2.0) ** 2 * anchor_w
        bh = (reg[..., 3] * 2.0) ** 2 * anchor_h

        cls_id = np.argmax(cls, axis=-1)
        cls_score = np.max(cls, axis=-1)
        conf = obj * cls_score[:, :, None]
        cls_id = np.repeat(cls_id[:, :, None], na, axis=2)

        boxes = np.stack([cx, cy, bw, bh], axis=-1).reshape(-1, 4)
        conf = conf.reshape(-1)
        cls_id = cls_id.reshape(-1)
        return boxes, conf, cls_id

    def postprocess(self, outputs, ratio_w, ratio_h):
        if len(outputs) != _NUM_SCALES:
            raise ValueError(
                "yolo_fastestv2: expected the ONNX model to have "
                f"{_NUM_SCALES} outputs (one per FPN scale), got "
                f"{len(outputs)}. This model does not look like a "
                "Yolo-FastestV2 export from pytorch2onnx.py."
            )

        all_boxes, all_conf, all_cls = [], [], []
        for scale_idx, out in enumerate(outputs):
            boxes, conf, cls_id = self._decode_scale(out, scale_idx)
            all_boxes.append(boxes)
            all_conf.append(conf)
            all_cls.append(cls_id)
        boxes = np.concatenate(all_boxes, axis=0)
        conf = np.concatenate(all_conf, axis=0)
        cls_id = np.concatenate(all_cls, axis=0)

        keep = conf > self.conf_thres
        boxes, conf, cls_id = boxes[keep], conf[keep], cls_id[keep]
        if self.filter_classes:
            filter_ids = {
                i
                for i, name in enumerate(self.classes)
                if name in self.filter_classes
            }
            keep = np.array([c in filter_ids for c in cls_id], dtype=bool)
            boxes, conf, cls_id = boxes[keep], conf[keep], cls_id[keep]

        output_infos = []
        if boxes.shape[0] == 0:
            return output_infos

        # cv2.dnn.NMSBoxes expects [x, y, w, h] with (x, y) = top-left
        nms_boxes = boxes.copy()
        nms_boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        nms_boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        indices = cv2.dnn.NMSBoxes(
            nms_boxes.tolist(),
            conf.tolist(),
            self.conf_thres,
            self.nms_thres,
        )
        for i in np.array(indices).reshape(-1):
            cx, cy, bw, bh = nms_boxes[i]
            xmin = cx * ratio_w
            ymin = cy * ratio_h
            xmax = (cx + bw) * ratio_w
            ymax = (cy + bh) * ratio_h
            output_infos.append(
                {
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "label": str(self.classes[int(cls_id[i])]),
                    "score": float(conf[i]),
                }
            )
        return output_infos

    def predict_shapes(self, image, image_path=None):
        """
        Predict shapes from image
        """

        if image is None:
            return []

        try:
            image = qt_img_to_rgb_cv_img(image, image_path)
        except Exception as e:  # noqa
            logger.warning("Could not inference model")
            logger.warning(e)
            return []

        blob, ratio_w, ratio_h = self.preprocess(image)
        outputs = self.net.get_ort_inference(blob, extract=False)
        results = self.postprocess(outputs, ratio_w, ratio_h)

        shapes = []
        for result in results:
            shape = Shape(
                label=result["label"],
                score=result["score"],
                shape_type="rectangle",
            )
            pt1 = QtCore.QPointF(result["xmin"], result["ymin"])
            pt2 = QtCore.QPointF(result["xmax"], result["ymin"])
            pt3 = QtCore.QPointF(result["xmax"], result["ymax"])
            pt4 = QtCore.QPointF(result["xmin"], result["ymax"])
            shape.add_point(pt1)
            shape.add_point(pt2)
            shape.add_point(pt3)
            shape.add_point(pt4)
            shapes.append(shape)
        result = AutoLabelingResult(shapes, replace=self.replace)
        return result

    def unload(self):
        del self.net
