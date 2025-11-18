import collections
import ctypes as ct
import ctypes.util
import json
import math
import os
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


def _find_library() -> Optional[str]:
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    search_path = [
        ".",
        os.path.join(os.path.dirname(__file__), "lib"),
        os.path.dirname(__file__),
    ] + ld_library_path
    for path in search_path:
        for name in ("libscanditsdk.so", "libscanditsdk.dylib"):
            fn = os.path.join(path, name)
            if os.path.exists(fn):
                return fn
    lib_path = ct.util.find_library("scanditsdk")
    if lib_path:
        return lib_path
    return None


_lib_path = _find_library()
if _lib_path is None:
    raise ImportError("could not find scanditsdk library.")

_ = ct.CDLL(_lib_path)


class NativeHandle(object):
    """Class for wrapping a handle to a _ object as returned by the scanditsdk C-API.

    This class is for internal use, and you won't typically interact with it directly.
    """

    def __init__(self, handle, release_func):
        self._as_parameter_ = ct.c_void_p(handle)
        self._release_func = release_func

    def dispose(self):
        self._release_func(self._as_parameter_)
        self._as_parameter_ = None

    @property
    def handle(self):
        return self._as_parameter_

    def __del__(self):
        self.dispose()


SYMBOLOGY_UNKNOWN = int(0)
SYMBOLOGY_EAN13_UPCA = int(1)
SYMBOLOGY_EAN8 = int(2)
SYMBOLOGY_UPCE = int(4)
SYMBOLOGY_CODE128 = int(5)
SYMBOLOGY_CODE39 = int(6)
SYMBOLOGY_CODE93 = int(7)
SYMBOLOGY_INTERLEAVED_2_OF_5 = int(8)
SYMBOLOGY_QR = int(9)
SYMBOLOGY_DATA_MATRIX = int(10)
SYMBOLOGY_PDF417 = int(11)
SYMBOLOGY_MSI_PLESSEY = int(12)
SYMBOLOGY_GS1_DATABAR = int(13)
SYMBOLOGY_GS1_DATABAR_EXPANDED = int(14)
SYMBOLOGY_CODABAR = int(15)
SYMBOLOGY_AZTEC = int(16)
SYMBOLOGY_TWO_DIGIT_ADD_ON = int(17)
SYMBOLOGY_FIVE_DIGIT_ADD_ON = int(18)
SYMBOLOGY_MAXICODE = int(19)
SYMBOLOGY_CODE11 = int(20)
SYMBOLOGY_GS1_DATABAR_LIMITED = int(21)
SYMBOLOGY_CODE25 = int(22)
SYMBOLOGY_MICRO_PDF417 = int(23)
SYMBOLOGY_RM4SCC = int(24)
SYMBOLOGY_KIX = int(24)
SYMBOLOGY_DOTCODE = int(26)
SYMBOLOGY_MICRO_QR = int(27)
SYMBOLOGY_CODE32 = int(28)
SYMBOLOGY_LAPA4SC = int(29)
SYMBOLOGY_IATA_2_OF_5 = int(30)
SYMBOLOGY_MATRIX_2_OF_5 = int(31)
SYMBOLOGY_INTELLIGENT_MAIL = int(32)
SYMBOLOGY_ARUCO = int(33)
SYMBOLOGY_UPU_4STATE = int(34)
SYMBOLOGY_AUSTRALIAN_POST_4STATE = int(35)
SYMBOLOGY_FRENCH_POST = int(36)

"""
List of all symbologies supported by the barcode scanner SDK.
"""
SYMBOLOGIES = (
    SYMBOLOGY_EAN13_UPCA,
    SYMBOLOGY_EAN8,
    SYMBOLOGY_UPCE,
    SYMBOLOGY_CODE128,
    SYMBOLOGY_CODE39,
    SYMBOLOGY_CODE93,
    SYMBOLOGY_INTERLEAVED_2_OF_5,
    SYMBOLOGY_QR,
    SYMBOLOGY_DATA_MATRIX,
    SYMBOLOGY_PDF417,
    SYMBOLOGY_MSI_PLESSEY,
    SYMBOLOGY_GS1_DATABAR,
    SYMBOLOGY_GS1_DATABAR_EXPANDED,
    SYMBOLOGY_CODABAR,
    SYMBOLOGY_AZTEC,
    SYMBOLOGY_TWO_DIGIT_ADD_ON,
    SYMBOLOGY_FIVE_DIGIT_ADD_ON,
    SYMBOLOGY_MAXICODE,
    SYMBOLOGY_CODE11,
    SYMBOLOGY_GS1_DATABAR_LIMITED,
    SYMBOLOGY_CODE25,
    SYMBOLOGY_MICRO_PDF417,
    SYMBOLOGY_RM4SCC,
    SYMBOLOGY_KIX,
    SYMBOLOGY_DOTCODE,
    SYMBOLOGY_MICRO_QR,
    SYMBOLOGY_CODE32,
    SYMBOLOGY_LAPA4SC,
    SYMBOLOGY_IATA_2_OF_5,
    SYMBOLOGY_MATRIX_2_OF_5,
    SYMBOLOGY_INTELLIGENT_MAIL,
)

COMPOSITE_FLAG_NONE = int(0)
COMPOSITE_FLAG_UNKNOWN = int(1)
COMPOSITE_FLAG_LINKED = int(2)
COMPOSITE_FLAG_GS1_A = int(3)
COMPOSITE_FLAG_GS1_B = int(4)
COMPOSITE_FLAG_GS1_C = int(5)


class Point(ct.Structure):
    """A two-dimensional point (x, y) with integer precision."""

    _fields_ = [("x", ct.c_int), ("y", ct.c_int)]

    def __getitem__(self, index: int) -> ct.c_int:
        if index == 0:
            return self.x
        if index == 1:
            return self.y
        return ct.c_int(-1)

    def __len__(self) -> int:
        return 2

    def __str__(self) -> str:
        return "({self.x}, {self.y})".format(self=self)

    def distance_to(self, other: "Point") -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


class PointF(ct.Structure):
    """A two-dimensional point (x, y) with floating-point precision."""

    _fields_ = [("x", ct.c_float), ("y", ct.c_float)]

    def __getitem__(self, index: int) -> ct.c_float:
        if index == 0:
            return self.x
        if index == 1:
            return self.y
        return ct.c_float(-1.0)

    def __len__(self) -> int:
        return 2


class Quadrilateral(ct.Structure):
    """A quadrilateral as defined by 4 corner points with integer precision."""

    _fields_ = [
        ("top_left", Point),
        ("top_right", Point),
        ("bottom_right", Point),
        ("bottom_left", Point),
    ]

    @property
    def all_corners(self):
        """Returns all the points in clock-wise order."""
        yield self.top_left
        yield self.top_right
        yield self.bottom_right
        yield self.bottom_left

    def __str__(self) -> str:
        return (
            "top_left={self.top_left}, top_right={self.top_right}, "
            "bottom_right={self.bottom_right}, bottom_left={self.bottom_left}".format(
                self=self
            )
        )


class FacingDirection(Enum):
    UNKNOWN = 0
    BACK = 1
    FRONT = 2


class _CameraProperties(ct.Structure):
    _fields_ = [("facing_direction", ct.c_int), ("identifier", ct.c_char_p)]


class ProcessFrameResult(ct.Structure):
    _fields_ = [("status", ct.c_int), ("frame_id", ct.c_uint)]

    def get_status_flag_message(self) -> str:
        return _.sc_context_status_flag_get_message(self.status).decode("utf-8")


class _DataPointer(ct.Union):
    _fields_ = [("str", ct.c_char_p), ("bytes", ct.POINTER(ct.c_uint8))]


class ByteArray(ct.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("u", _DataPointer), ("size", ct.c_uint32), ("flags", ct.c_uint32)]


class _EncodingRange(ct.Structure):
    _fields_ = [
        ("encoding", ByteArray),
        ("start", ct.c_uint),
        ("end", ct.c_uint),
    ]


class _EncodingRangeArray(ct.Structure):
    _fields_ = [("encodings", ct.POINTER(_EncodingRange)), ("size", ct.c_uint)]


class _Error(ct.Structure):
    _fields_ = [("message", ct.c_char_p), ("code", ct.c_uint)]


EncodingRange = collections.namedtuple("EncodingRange", "encoding start end")


CODE_DIRECTION_NONE = 0x00
CODE_DIRECTION_LEFT_TO_RIGHT = 0x01
CODE_DIRECTION_RIGHT_TO_LEFT = 0x02
CODE_DIRECTION_TOP_TO_BOTTOM = 0x04
CODE_DIRECTION_BOTTOM_TO_TOP = 0x08
CODE_DIRECTION_VERTICAL = CODE_DIRECTION_TOP_TO_BOTTOM | CODE_DIRECTION_BOTTOM_TO_TOP
CODE_DIRECTION_HORIZONTAL = CODE_DIRECTION_LEFT_TO_RIGHT | CODE_DIRECTION_RIGHT_TO_LEFT

PRESET_NONE = 0x00
PRESET_HIGH_EFFORT = 0x02
PRESET_SINGLE_CODE_HAND_HELD = 0x08


class _ScPointF(ct.Structure):
    _fields_ = [("x", ct.c_float), ("y", ct.c_float)]


class _ScSizeF(ct.Structure):
    _fields_ = [("width", ct.c_float), ("height", ct.c_float)]


class _ScRectangleF(ct.Structure):
    _fields_ = [("position", _ScPointF), ("size", _ScSizeF)]

    @staticmethod
    def from_position_and_size(
        position: Tuple[float, float], size: Tuple[float, float]
    ) -> "_ScRectangleF":
        location = _ScRectangleF()
        location.position.x = ct.c_float(position[0])
        location.position.y = ct.c_float(position[1])
        location.size.width = ct.c_float(size[0])
        location.size.height = ct.c_float(size[1])
        return location

    def toList(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return (
            (float(self.position.x), float(self.position.y)),
            (float(self.size.width), float(self.size.height)),
        )


_.sc_property_collection_is_property_known.argtypes = [ct.c_void_p, ct.c_char_p]
_.sc_property_collection_get_bool_property.argtypes = [
    ct.c_void_p,
    ct.c_char_p,
    ct.POINTER(ct.c_int),
]
_.sc_property_collection_get_int_property.argtypes = [
    ct.c_void_p,
    ct.c_char_p,
    ct.POINTER(ct.c_int),
]
_.sc_property_collection_get_float_property.argtypes = [
    ct.c_void_p,
    ct.c_char_p,
    ct.POINTER(ct.c_float),
]
_.sc_property_collection_get_string_property.argtypes = [
    ct.c_void_p,
    ct.c_char_p,
    ct.POINTER(ct.c_char_p),
]
_.sc_property_collection_set_bool_property.argtypes = [
    ct.c_void_p,
    ct.c_char_p,
    ct.c_int,
]
_.sc_property_collection_set_int_property.argtypes = [
    ct.c_void_p,
    ct.c_char_p,
    ct.c_int,
]
_.sc_property_collection_set_float_property.argtypes = [
    ct.c_void_p,
    ct.c_char_p,
    ct.c_float,
]
_.sc_property_collection_set_string_property.argtypes = [
    ct.c_void_p,
    ct.c_char_p,
    ct.c_char_p,
]


class PropertyCollection(NativeHandle):
    def __init__(self, handle: ct.c_void_p):
        NativeHandle.__init__(self, handle, lambda _handle: None)

    def is_property_known(self, name: str) -> bool:
        return (
            True
            if _.sc_property_collection_is_property_known(
                self, ct.c_char_p(name.encode("utf-8"))
            )
            == 1
            else False
        )

    def get_bool_property(self, name: str) -> Optional[bool]:
        prop_value = ct.c_int()
        if (
            _.sc_property_collection_get_bool_property(
                self, ct.c_char_p(name.encode("utf-8")), ct.byref(prop_value)
            )
            == 1
        ):
            return True if prop_value.value == 1 else False
        return None

    def get_int_property(self, name: str) -> Optional[int]:
        prop_value = ct.c_int()
        if (
            _.sc_property_collection_get_int_property(
                self, ct.c_char_p(name.encode("utf-8")), ct.byref(prop_value)
            )
            == 1
        ):
            return prop_value.value
        return None

    def get_float_property(self, name: str) -> Optional[float]:
        prop_value = ct.c_float()
        if (
            _.sc_property_collection_get_float_property(
                self, ct.c_char_p(name.encode("utf-8")), ct.byref(prop_value)
            )
            == 1
        ):
            return prop_value.value
        return None

    def get_string_property(self, name: str) -> Optional[float]:
        prop_value = ct.c_char_p()
        if (
            _.sc_property_collection_get_string_property(
                self, ct.c_char_p(name.encode("utf-8")), ct.byref(prop_value)
            )
            == 1
        ):
            return prop_value.value.decode("utf-8")
        return None

    def set_bool_property(self, name: str, value: bool):
        if (
            _.sc_property_collection_set_bool_property(
                self, ct.c_char_p(name.encode("utf-8")), ct.c_int(value)
            )
            == 0
        ):
            raise RuntimeError(
                f"Failed to set bool value `{value}` to property `{name}`"
            )

    def set_int_property(self, name: str, value: int):
        if (
            _.sc_property_collection_set_int_property(
                self, ct.c_char_p(name.encode("utf-8")), ct.c_int(value)
            )
            == 0
        ):
            raise RuntimeError(
                f"Failed to set int value `{value}` to property `{name}`"
            )

    def set_float_property(self, name: str, value: float):
        if (
            _.sc_property_collection_set_float_property(
                self, ct.c_char_p(name.encode("utf-8")), ct.c_float(value)
            )
            == 0
        ):
            raise RuntimeError(
                f"Failed to set float value `{value}` to property `{name}`"
            )

    def set_string_property(self, name: str, value: str):
        if (
            _.sc_property_collection_set_string_property(
                self,
                ct.c_char_p(name.encode("utf-8")),
                ct.c_char_p(value.encode("utf-8")),
            )
            == 0
        ):
            raise RuntimeError(
                f"Failed to set string value `{value}` to property `{name}`"
            )


_.sc_barcode_scanner_settings_as_json.restype = ct.c_char_p
_.sc_barcode_scanner_settings_as_json.argtypes = [ct.c_void_p]


_.sc_barcode_scanner_settings_new_with_preset.restype = ct.c_void_p
_.sc_barcode_scanner_settings_new_with_preset.argtypes = [ct.c_int]
_.sc_barcode_scanner_settings_new_from_json.restype = ct.c_void_p
_.sc_barcode_scanner_settings_new_from_json.argtypes = [ct.c_char_p, ct.POINTER(_Error)]
_.sc_barcode_scanner_settings_set_symbology_enabled.argtypes = [
    ct.c_void_p,
    ct.c_int,
    ct.c_int,
]
_.sc_barcode_scanner_settings_retain.argtypes = [ct.c_void_p]
_.sc_barcode_scanner_settings_release.argtypes = [ct.c_void_p]

_.sc_barcode_scanner_settings_get_properties.restype = ct.c_void_p
_.sc_barcode_scanner_settings_get_properties.argtypes = [ct.c_void_p]

_.sc_barcode_scanner_settings_set_code_duplicate_filter.argtypes = [
    ct.c_void_p,
    ct.c_int,
]
_.sc_barcode_scanner_settings_get_code_duplicate_filter.restype = ct.c_int
_.sc_barcode_scanner_settings_get_code_duplicate_filter.argtypes = [ct.c_void_p]

_.sc_barcode_scanner_settings_set_code_direction_hint.argtypes = [ct.c_void_p, ct.c_int]
_.sc_barcode_scanner_settings_get_code_direction_hint.restype = ct.c_int
_.sc_barcode_scanner_settings_get_code_direction_hint.argtypes = [ct.c_void_p]
_.sc_barcode_scanner_settings_set_max_number_of_codes_per_frame.argtypes = [
    ct.c_void_p,
    ct.c_int,
]
_.sc_barcode_scanner_settings_get_max_number_of_codes_per_frame.restype = ct.c_int
_.sc_barcode_scanner_settings_get_max_number_of_codes_per_frame.argtypes = [ct.c_void_p]
_.sc_barcode_scanner_settings_set_code_location_area_1d.argtypes = [
    ct.c_void_p,
    _ScRectangleF,
]
_.sc_barcode_scanner_settings_set_code_location_constraint_1d.argtypes = [
    ct.c_void_p,
    ct.c_int,
]
_.sc_barcode_scanner_settings_get_code_location_constraint_1d.restype = ct.c_int
_.sc_barcode_scanner_settings_set_code_location_constraint_2d.argtypes = [
    ct.c_void_p,
    ct.c_int,
]
_.sc_barcode_scanner_settings_get_code_location_constraint_2d.restype = ct.c_int

_.sc_barcode_scanner_settings_get_code_location_area_2d.argtypes = [ct.c_void_p]
_.sc_barcode_scanner_settings_get_code_location_area_2d.restype = _ScRectangleF
_.sc_barcode_scanner_settings_get_code_location_area_1d.argtypes = [ct.c_void_p]
_.sc_barcode_scanner_settings_get_code_location_area_1d.restype = _ScRectangleF

_.sc_barcode_scanner_settings_set_code_location_area_2d.argtypes = [
    ct.c_void_p,
    _ScRectangleF,
]
_.sc_barcode_scanner_settings_set_search_area.argtypes = [
    ct.c_void_p,
    _ScRectangleF,
]
_.sc_barcode_scanner_settings_get_search_area.argtypes = [ct.c_void_p]
_.sc_barcode_scanner_settings_get_search_area.restype = _ScRectangleF

_.sc_barcode_scanner_settings_set_focus_mode.argtypes = [ct.c_void_p, ct.c_int]
_.sc_barcode_scanner_settings_get_focus_mode.argtypes = [ct.c_void_p]
_.sc_barcode_scanner_settings_get_focus_mode.restype = ct.c_int


_.sc_barcode_scanner_settings_get_symbology_settings.argtypes = [ct.c_void_p, ct.c_int]
_.sc_barcode_scanner_settings_get_symbology_settings.restype = ct.c_void_p
_.sc_symbology_settings_retain.argtypes = [ct.c_void_p]
_.sc_symbology_settings_release.argtypes = [ct.c_void_p]
_.sc_symbology_settings_get_active_symbol_counts.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.POINTER(ct.c_uint16),
]
_.sc_symbology_settings_set_active_symbol_counts.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_uint16),
    ct.c_uint16,
]

_.sc_barcode_scanner_new_with_settings.argtypes = [ct.c_void_p, ct.c_void_p]
_.sc_barcode_scanner_new_with_settings.restype = ct.c_void_p
_.sc_barcode_scanner_retain.argtypes = [ct.c_void_p]
_.sc_barcode_scanner_release.argtypes = [ct.c_void_p]
_.sc_barcode_scanner_wait_for_setup_completed.argtypes = [ct.c_void_p]
_.sc_barcode_scanner_wait_for_setup_completed.restype = ct.c_int

_.sc_recognition_context_new.restype = ct.c_void_p
_.sc_recognition_context_new.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_char_p]
_.sc_recognition_context_retain.argtypes = [ct.c_void_p]
_.sc_recognition_context_release.argtypes = [ct.c_void_p]
_.sc_recognition_context_start_new_frame_sequence.argtypes = [ct.c_void_p]
_.sc_recognition_context_end_frame_sequence.argtypes = [ct.c_void_p]
_.sc_recognition_context_process_frame.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
]
_.sc_recognition_context_process_frame.restype = ProcessFrameResult
_.sc_recognition_context_set_camera_properties.argtypes = [
    ct.c_void_p,
    _CameraProperties,
]

_.sc_context_status_flag_get_message.argtypes = [ct.c_int]
_.sc_context_status_flag_get_message.restype = ct.c_char_p

IMAGE_LAYOUT_UNKNOWN = ct.c_int(0x0000)
IMAGE_LAYOUT_GRAY_8U = ct.c_int(0x0001)
IMAGE_LAYOUT_RGB_8U = ct.c_int(0x0002)
IMAGE_LAYOUT_RGBA_8U = ct.c_int(0x0004)
IMAGE_LAYOUT_ARGB_8U = ct.c_int(0x0100)
IMAGE_LAYOUT_YPCBCR_8U = ct.c_int(0x0008)
IMAGE_LAYOUT_YPCRCB_8U = ct.c_int(0x0010)
IMAGE_LAYOUT_YUYV_8U = ct.c_int(0x0020)
IMAGE_LAYOUT_UYVY_8U = ct.c_int(0x004)
IMAGE_LAYOUT_BGR_8U = ct.c_int(0x0200)
IMAGE_LAYOUT_NV16_U8 = ct.c_int(0x0800)
IMAGE_LAYOUT_I420_8U = ct.c_int(0x0080)

CODE_LOCATION_RESTRICT = 0x01
CODE_LOCATION_HINT = 0x02
CODE_LOCATION_IGNORE = 0x03

_.sc_barcode_scanner_get_session.argtypes = [ct.c_void_p]
_.sc_barcode_scanner_get_session.restype = ct.c_void_p

_.sc_barcode_scanner_session_retain.argtypes = [ct.c_void_p]
_.sc_barcode_scanner_session_release.argtypes = [ct.c_void_p]

_.sc_barcode_scanner_session_get_newly_recognized_codes.argtypes = [ct.c_void_p]
_.sc_barcode_scanner_session_get_newly_recognized_codes.restype = ct.c_void_p

_.sc_barcode_scanner_session_get_all_recognized_codes.argtypes = [ct.c_void_p]
_.sc_barcode_scanner_session_get_all_recognized_codes.restype = ct.c_void_p

_.sc_barcode_scanner_session_get_newly_localized_codes.argtypes = [ct.c_void_p]
_.sc_barcode_scanner_session_get_newly_localized_codes.restype = ct.c_void_p

_.sc_object_tracker_new.argtypes = [ct.c_void_p]
_.sc_object_tracker_new.restype = ct.c_void_p
_.sc_object_tracker_retain.argtypes = [ct.c_void_p]
_.sc_object_tracker_release.argtypes = [ct.c_void_p]
_.sc_object_tracker_apply_scanner_settings.argtypes = [ct.c_void_p, ct.c_void_p]
_.sc_object_tracker_apply_scanner_settings.restype = ct.c_void_p

_.sc_tracked_object_retain.argtypes = [ct.c_void_p]
_.sc_tracked_object_release.argtypes = [ct.c_void_p]
_.sc_tracked_object_get_barcode.argtypes = [ct.c_void_p]
_.sc_tracked_object_get_barcode.restype = ct.c_void_p
_.sc_tracked_object_get_location.argtypes = [ct.c_void_p]
_.sc_tracked_object_get_location.restype = Quadrilateral
_.sc_tracked_object_get_id.argtypes = [ct.c_void_p]
_.sc_tracked_object_get_id.restype = ct.c_uint32

_.sc_object_tracker_get_session.argtypes = [ct.c_void_p]
_.sc_object_tracker_get_session.restype = ct.c_void_p
_.sc_object_tracker_session_get_tracked_objects.argtypes = [ct.c_void_p]
_.sc_object_tracker_session_get_tracked_objects.restype = ct.c_void_p

_.sc_tracked_object_map_release.argtypes = [ct.c_void_p]
_.sc_tracked_object_map_get_size.argtypes = [ct.c_void_p]
_.sc_tracked_object_map_get_size.restype = ct.c_uint
_.sc_tracked_object_map_get_ids.argtypes = [ct.c_void_p]
_.sc_tracked_object_map_get_ids.restype = ct.POINTER(ct.c_uint)
_.sc_tracked_object_map_get_item_at.argtypes = [ct.c_void_p, ct.c_uint]
_.sc_tracked_object_map_get_item_at.restype = ct.c_void_p

_.sc_image_description_new.restype = ct.c_void_p
_.sc_image_description_retain.argtypes = [ct.c_void_p]
_.sc_image_description_release.argtypes = [ct.c_void_p]
_.sc_image_description_set_width.argtypes = [ct.c_void_p, ct.c_uint32]
_.sc_image_description_set_height.argtypes = [ct.c_void_p, ct.c_uint32]
_.sc_image_description_set_layout.argtypes = [ct.c_void_p, ct.c_int]
_.sc_image_description_set_plane_offset.argtypes = [
    ct.c_void_p,
    ct.c_uint32,
    ct.c_ssize_t,
]
_.sc_image_description_set_plane_row_bytes.argtypes = [
    ct.c_void_p,
    ct.c_uint32,
    ct.c_uint32,
]

_.sc_image_description_get_width.argtypes = [ct.c_void_p]
_.sc_image_description_get_width.restype = ct.c_uint32
_.sc_image_description_get_height.argtypes = [ct.c_void_p]
_.sc_image_description_get_height.restype = ct.c_uint32
_.sc_image_description_get_layout.argtypes = [ct.c_void_p]
_.sc_image_description_get_layout.restype = ct.c_int
_.sc_image_description_get_planes_count.argtypes = [ct.c_void_p]
_.sc_image_description_get_planes_count.restype = ct.c_uint32
_.sc_image_description_get_plane_offset.argtypes = [ct.c_void_p, ct.c_uint32]
_.sc_image_description_get_plane_offset.restype = ct.c_ssize_t
_.sc_image_description_get_plane_row_bytes.argtypes = [ct.c_void_p, ct.c_uint32]
_.sc_image_description_get_plane_row_bytes.restype = ct.c_uint32

_.sc_barcode_array_retain.argtypes = [ct.c_void_p]
_.sc_barcode_array_release.argtypes = [ct.c_void_p]
_.sc_barcode_array_get_item_at.argtypes = [ct.c_void_p, ct.c_uint32]
_.sc_barcode_array_get_item_at.restype = ct.c_void_p
_.sc_barcode_array_get_size.argtypes = [ct.c_void_p]
_.sc_barcode_array_get_size.restype = ct.c_int

_.sc_barcode_retain.argtypes = [ct.c_void_p]
_.sc_barcode_get_data.argtypes = [ct.c_void_p]
_.sc_barcode_get_data.restype = ByteArray
_.sc_barcode_get_location.argtypes = [ct.c_void_p]
_.sc_barcode_get_location.restype = Quadrilateral
_.sc_barcode_get_symbology.argtypes = [ct.c_void_p]
_.sc_barcode_get_symbology.restype = ct.c_int
_.sc_symbology_to_string.argtypes = [ct.c_int]
_.sc_symbology_to_string.restype = ct.c_char_p
_.sc_symbology_from_string.argtypes = [ct.c_char_p]
_.sc_symbology_from_string.restype = ct.c_int

_.sc_barcode_release.argtypes = [ct.c_void_p]
_.sc_barcode_get_data_encoding.argtypes = [ct.c_void_p]
_.sc_barcode_get_data_encoding.restype = _EncodingRangeArray

_.sc_encoding_array_new.argtypes = [ct.c_uint32]
_.sc_encoding_array_new.restype = _EncodingRangeArray
_.sc_encoding_array_free.argtypes = [_EncodingRangeArray]

_.sc_error_free.argtypes = [ct.POINTER(_Error)]


class _ImageBuffer(ct.Structure):
    _fields_ = [
        ("description", ct.c_void_p),
        ("data", ct.c_void_p),
        ("data_size", ct.c_size_t),
    ]


_.sc_barcode_generator_new.argtypes = [ct.c_void_p, ct.c_int, ct.POINTER(_Error)]
_.sc_barcode_generator_new.restype = ct.c_void_p
_.sc_barcode_generator_set_options.argtypes = [
    ct.c_void_p,
    ct.c_char_p,
    ct.POINTER(_Error),
]
_.sc_barcode_generator_set_options.restype = ct.c_void_p
_.sc_barcode_generator_free.argtypes = [ct.c_void_p]
_.sc_barcode_generator_free.restype = ct.c_void_p
_.sc_barcode_generator_generate.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_size_t,
    _EncodingRangeArray,
    ct.POINTER(_Error),
]
_.sc_barcode_generator_generate.restype = ct.POINTER(_ImageBuffer)

_.sc_image_buffer_free.argtypes = [ct.c_void_p]
_.sc_image_buffer_free.restype = ct.c_void_p

RECOGNITION_CONTEXT_STATUS_UNKNOWN = 0
RECOGNITION_CONTEXT_STATUS_SUCCESS = 1
RECOGNITION_CONTEXT_STATUS_INTERNAL_ERROR = 2
RECOGNITION_CONTEXT_STATUS_FRAME_SEQUENCE_NOT_STARTED = 3
RECOGNITION_CONTEXT_STATUS_UNSUPPORTED_IMAGE_DATA = 4
RECOGNITION_CONTEXT_STATUS_INCONSISTENT_IMAGE_DATA = 5
RECOGNITION_CONTEXT_STATUS_NO_NETWORK_CONNECTION = 6
RECOGNITION_CONTEXT_STATUS_LICENSE_VALIDATION_FAILED_403 = 11
RECOGNITION_CONTEXT_STATUS_LICENSE_KEY_MISSING = 12
RECOGNITION_CONTEXT_STATUS_LICENSE_KEY_EXPIRED = 13
RECOGNITION_CONTEXT_STATUS_INVALID_PLATFORM = 14
RECOGNITION_CONTEXT_STATUS_INVALID_APP_ID = 15
RECOGNITION_CONTEXT_STATUS_INVALID_DEVICE = 16
RECOGNITION_CONTEXT_STATUS_INVALID_SDK_VERSION = 17
RECOGNITION_CONTEXT_STATUS_LICENSE_KEY_INVALID = 18
RECOGNITION_CONTEXT_STATUS_DEVICE_ACTIVATION_FAILED = 19
RECOGNITION_CONTEXT_STATUS_TIME_EXCEEDED = 20
RECOGNITION_CONTEXT_STATUS_SCANS_EXCEEDED = 21
RECOGNITION_CONTEXT_STATUS_REGISTRATION_MANDATORY_BUT_NOT_REGISTERED = 22
RECOGNITION_CONTEXT_STATUS_INVALID_EXTERNAL_ID = 23
RECOGNITION_CONTEXT_STATUS_UNLICENSED_SYMBOLOGY_ENABLED = 24
RECOGNITION_CONTEXT_STATUS_INVALID_LICENSE_KEY_VERSION = 26
RECOGNITION_CONTEXT_STATUS_UNLICENSED_FEATURE_BEGIN = 255
RECOGNITION_CONTEXT_STATUS_UNLICENSED_FEATURE_END = 512


class _ScStepwiseResolution(ct.Structure):
    _fields_ = [
        ("min_width", ct.c_uint32),
        ("max_width", ct.c_uint32),
        ("step_width", ct.c_uint32),
        ("min_height", ct.c_uint32),
        ("max_height", ct.c_uint32),
        ("step_height", ct.c_uint32),
    ]


class _ScFramerate(ct.Structure):
    _fields_ = [("numerator", ct.c_uint32), ("denominator", ct.c_uint32)]


class _ScSize(ct.Structure):
    _fields_ = [("width", ct.c_uint32), ("height", ct.c_uint32)]


class _ScStepwiseFramerate(ct.Structure):
    _fields_ = [("min", ct.c_uint32), ("max", ct.c_uint32), ("step", ct.c_uint32)]


CAMERA_MODE_UNKNOWN = 0
CAMERA_MODE_DISCRETE = 1
CAMERA_MODE_STEPWISE = 2

CAMERA_FOCUS_MODE_UNKNOWN = 0
CAMERA_FOCUS_MODE_FIXED = 1
CAMERA_FOCUS_MODE_AUTO = 2
CAMERA_FOCUS_MODE_MANUAL = 4


_.sc_framerate_get_frame_interval.argtypes = [ct.POINTER(_ScFramerate)]
_.sc_framerate_get_frame_interval.restype = ct.c_float
_.sc_framerate_get_fps.argtypes = [ct.POINTER(_ScFramerate)]
_.sc_framerate_get_fps.restype = ct.c_float
_.sc_camera_new.argtypes = []
_.sc_camera_new.restype = ct.c_void_p
_.sc_camera_new_with_buffer_count.argtypes = [ct.c_uint32]
_.sc_camera_new_with_buffer_count.restype = ct.c_void_p
_.sc_camera_new_from_path.argtypes = [ct.c_char_p, ct.c_uint32]
_.sc_camera_new_from_path.restype = ct.c_void_p
_.sc_camera_release.argtypes = [ct.c_void_p]
_.sc_camera_release.restype = None
_.sc_camera_retain.argtypes = [ct.c_void_p]
_.sc_camera_retain.restype = None
_.sc_camera_get_resolution.argtypes = [ct.c_void_p]
_.sc_camera_get_resolution.restype = _ScSize
_.sc_camera_get_image_layout.argtypes = [ct.c_void_p]
_.sc_camera_get_image_layout.restype = ct.c_int32
_.sc_camera_get_resolution_mode.argtypes = [ct.c_void_p]
_.sc_camera_get_resolution_mode.restype = ct.c_int32
_.sc_camera_get_framerate_mode.argtypes = [ct.c_void_p]
_.sc_camera_get_framerate_mode.restype = ct.c_int32
_.sc_camera_query_supported_resolutions.argtypes = [
    ct.c_void_p,
    ct.POINTER(_ScSize),
    ct.c_uint32,
]
_.sc_camera_query_supported_resolutions.restype = ct.c_int32
_.sc_camera_query_supported_resolutions_stepwise.argtypes = [
    ct.c_void_p,
    ct.POINTER(_ScStepwiseResolution),
]
_.sc_camera_query_supported_resolutions_stepwise.restype = ct.c_int32
_.sc_camera_query_supported_framerates.argtypes = [
    ct.c_void_p,
    _ScSize,
    ct.POINTER(_ScFramerate),
    ct.c_int32,
]
_.sc_camera_query_supported_framerates.restype = ct.c_int32
_.sc_camera_query_supported_framerates_stepwise.argtypes = [
    ct.c_void_p,
    _ScSize,
    ct.POINTER(_ScStepwiseFramerate),
]
_.sc_camera_query_supported_framerates_stepwise.restype = ct.c_int32
_.sc_camera_request_resolution.argtypes = [ct.c_void_p, _ScSize]
_.sc_camera_request_resolution.restype = ct.c_int32
_.sc_camera_set_focus_mode.argtypes = [ct.c_void_p, ct.c_int32]
_.sc_camera_set_focus_mode.restype = ct.c_int32
_.sc_camera_set_manual_auto_focus_distance.argtypes = [ct.c_void_p, ct.c_int32]
_.sc_camera_set_manual_auto_focus_distance.restype = ct.c_int32
_.sc_camera_get_frame.argtypes = [ct.c_void_p, ct.c_void_p]
_.sc_camera_get_frame.restype = ct.POINTER(ct.c_uint8)
_.sc_camera_start_stream.argtypes = [ct.c_void_p]
_.sc_camera_start_stream.restype = ct.c_int32
_.sc_camera_stop_stream.argtypes = [ct.c_void_p]
_.sc_camera_stop_stream.restype = ct.c_int32
_.sc_camera_enqueue_frame_data.argtypes = [ct.c_void_p, ct.POINTER(ct.c_uint8)]
_.sc_camera_enqueue_frame_data.restype = ct.c_int32


class StepwiseResolution:
    def __init__(self, sc_stepwise_resolution: Optional[_ScStepwiseResolution]):
        self.min_width = ct.c_uint32(0)
        self.max_width = ct.c_uint32(0)
        self.step_width = ct.c_uint32(0)
        self.min_height = ct.c_uint32(0)
        self.max_height = ct.c_uint32(0)
        self.step_height = ct.c_uint32(0)
        if sc_stepwise_resolution is not None:
            self.min_width = sc_stepwise_resolution.min_width
            self.max_width = sc_stepwise_resolution.max_width
            self.step_width = sc_stepwise_resolution.step_width
            self.min_height = sc_stepwise_resolution.min_height
            self.max_height = sc_stepwise_resolution.max_height
            self.step_height = sc_stepwise_resolution.step_height


class SupportedResolutions:
    def __init__(self, camera: "Camera"):
        resolution_mode = _.sc_camera_get_resolution_mode(camera.handle)
        self.__discrete_resolutions = None
        self.__stepwise_resolutions = None

        if resolution_mode == CAMERA_MODE_DISCRETE:
            resolutions_size = 30
            resolutions = (_ScSize * resolutions_size)()
            resolutions_ptr = ct.cast(resolutions, ct.POINTER(_ScSize))
            _.sc_camera_query_supported_resolutions(
                camera.handle, resolutions_ptr, resolutions_size
            )
            self.__discrete_resolutions = self.__filter_available_resolutions(
                resolutions
            )
        elif resolution_mode == CAMERA_MODE_STEPWISE:
            stepwise_resolutions = _ScStepwiseResolution()

            _.sc_camera_query_supported_resolutions_stepwise(
                camera.handle, ct.byref(stepwise_resolutions)
            )
            self.__stepwise_resolutions = StepwiseResolution(stepwise_resolutions)
        else:
            raise RuntimeError(f"Unknown camera resolution mode: {resolution_mode}")

    @property
    def stepwise_resolutions(self) -> Optional[StepwiseResolution]:
        return self.__stepwise_resolutions

    @property
    def discrete_resolutions(self) -> Optional[List[_ScSize]]:
        return self.__discrete_resolutions

    def __filter_available_resolutions(
        self, resolutions
    ) -> List[_ScSize]:
        valid_resolutions: list[_ScSize] = list()
        for resolution in resolutions:
            if resolution.width == 0:
                break
            valid_resolutions.append(resolution)
        return valid_resolutions


class CameraFrame:
    # TYPE_CHECKING is True if `mypy` is run and False otherwise. `mypy` allows the type of
    # the pointer to be specified while `pytest` reports an error when doing so. This if-statement
    # makes both happy.
    # Using a Union is a hack to tell `mypy` that `DataT` is a type and not a variable. Without
    # the Union, `DataT` is considered as being a variable and it cannot be used to specify the
    # return type for property `data`.
    if TYPE_CHECKING:
        DataT = Union[ct.POINTER(ct.c_uint8)]
    else:
        DataT = Union[ct.pointer]

    def __init__(self, camera: "Camera"):
        self.__camera_handle = camera.handle
        self.__description = ImageDescription()
        self.__data = None
        image_data = _.sc_camera_get_frame(camera.handle, self.__description.handle)
        if not image_data:
            raise RuntimeError("Failed to retrieve data from the camera")
        self.__data = ct.cast(image_data, ct.POINTER(ct.c_uint8))

    @property
    def data(self) -> DataT:
        return self.__data

    @property
    def description(self) -> "ImageDescription":
        return self.__description

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback
        if self.__data:
            _.sc_camera_enqueue_frame_data(self.__camera_handle, self.__data)


class Camera(NativeHandle):
    def __init__(self, path: str = None, buffer_count: int = 4):
        if path is None:
            handle = _.sc_camera_new()
        elif type(path) is str and type(buffer_count) is int:
            handle = _.sc_camera_new_from_path(
                ct.c_char_p(path.encode("utf-8")), ct.c_uint32(buffer_count)
            )
        else:
            raise RuntimeError(
                "Camera path must be a string and buffer count must be an integer"
            )

        if handle is None:
            raise RuntimeError("No camera available")

        NativeHandle.__init__(self, handle, _.sc_camera_release)
        _.sc_camera_retain(self)

    @property
    def resolution(self) -> Tuple[int, int]:
        resolution = _.sc_camera_get_resolution(self)
        return resolution.width, resolution.height

    @resolution.setter
    def resolution(self, resolution: Tuple[int, int]):
        size = _ScSize(resolution[0], resolution[1])
        if not _.sc_camera_request_resolution(self, size):
            raise RuntimeError("Failed to set requested camera resolution")

    @property
    def supported_resolutions(self) -> SupportedResolutions:
        return SupportedResolutions(self)

    def start_streaming(self):
        if not _.sc_camera_start_stream(self):
            raise RuntimeError("Failed to start camera stream")

    @property
    def frame(self) -> CameraFrame:
        return CameraFrame(self)


class SymbologySettings(NativeHandle):
    """Holds symbology-specific settings."""

    def __init__(self, handle: ct.c_void_p):
        NativeHandle.__init__(self, handle, _.sc_symbology_settings_release)
        _.sc_symbology_settings_retain(self)

    @property
    def enabled(self) -> bool:
        return bool(_.sc_symbology_settings_is_enabled(self))

    @enabled.setter
    def enabled(self, is_enabled: bool):
        _.sc_symbology_settings_set_enabled(self, ct.c_int(is_enabled))

    @property
    def color_inverted_enabled(self) -> bool:
        return bool(_.sc_symbology_settings_is_color_inverted_enabled(self))

    @color_inverted_enabled.setter
    def color_inverted_enabled(self, is_enabled: bool):
        _.sc_symbology_settings_set_color_inverted_enabled(self, ct.c_int(is_enabled))

    def set_extension_enabled(self, name: str, is_enabled: bool):
        name_char_p = ct.c_char_p(name.encode("utf-8"))
        _.sc_symbology_settings_set_extension_enabled(
            self, name_char_p, ct.c_int(is_enabled)
        )

    def is_extension_enabled(self, name: str) -> bool:
        name_char_p = ct.c_char_p(name.encode("utf-8"))
        return bool(_.sc_symbology_settings_is_extension_enabled(self, name_char_p))

    @property
    def active_symbol_counts(self) -> List[ct.c_uint16]:
        the_data = ct.POINTER(ct.c_uint16)()
        count = ct.c_uint16(0)
        _.sc_symbology_settings_get_active_symbol_counts(
            self, ct.byref(the_data), ct.byref(count)
        )

        result = [the_data[i] for i in range(count.value)]
        _.sc_free(the_data)
        return result

    @active_symbol_counts.setter
    def active_symbol_counts(self, values: List[ct.c_uint16]):
        num_counts = len(values)
        # allocate buffer to hold the active symbol counts sizeof(uint16) * num_counts
        data_buffer = ct.create_string_buffer(num_counts * 2)
        data_buffer_ptr = ct.POINTER(type(data_buffer))(data_buffer)
        the_data = ct.cast(data_buffer_ptr, ct.POINTER(ct.c_uint16))
        for i, val in enumerate(values):
            the_data[i] = val
        _.sc_symbology_settings_set_active_symbol_counts(
            self, the_data, ct.c_uint16(num_counts)
        )

    @staticmethod
    def identifier_to_string(symbology_identifier: int) -> Optional[str]:
        """
        :return: The string representation for the specified symbology identifier or None if the symbology identifier
                 is invalid.
        """
        if symbology_identifier not in SYMBOLOGIES:
            return None
        return _.sc_symbology_to_string(symbology_identifier).decode("utf-8")


class BarcodeScannerSettings(NativeHandle):
    """Settings to control the scanning process."""

    def __init__(
        self, preset: Optional[int] = PRESET_NONE, handle: Optional[NativeHandle] = None
    ):
        if not handle:
            handle = _.sc_barcode_scanner_settings_new_with_preset(preset)
        NativeHandle.__init__(self, handle, _.sc_barcode_scanner_settings_release)

        def get_symbology_settings(s: int) -> SymbologySettings:
            h = _.sc_barcode_scanner_settings_get_symbology_settings(self, ct.c_int(s))
            return SymbologySettings(h)

        self.symbologies = {sym: get_symbology_settings(sym) for sym in SYMBOLOGIES}

    @staticmethod
    def from_json(string: str) -> "BarcodeScannerSettings":
        error = _Error()
        h = _.sc_barcode_scanner_settings_new_from_json(string.encode("utf-8"), error)
        if not h:
            message = error.message.decode("utf-8")
            _.sc_error_free(error)
            raise RuntimeError(message)
        return BarcodeScannerSettings(handle=h)

    @staticmethod
    def as_json(settings: "BarcodeScannerSettings"):
        h = _.sc_barcode_scanner_settings_as_json(settings)
        if not h:
            raise RuntimeError("Can't get json from settings")
        return h.decode("utf-8")

    def enable_symbology(
        self, symbology: Union[str, int], enable: Optional[bool] = True
    ):
        if type(symbology) is str:
            symbology_tmp = _.sc_symbology_from_string(symbology.encode("utf-8"))

            if symbology_tmp == 0:
                raise ValueError("Symbology does not exist: {}".format(symbology))
            symbology_enum = symbology_tmp
        elif type(symbology) is int:
            symbology_enum = ct.c_int(symbology)

        do_enable = ct.c_int(1) if (enable is None) or enable else ct.c_int(0)
        _.sc_barcode_scanner_settings_set_symbology_enabled(
            self, symbology_enum, do_enable
        )

    @property
    def properties(self) -> PropertyCollection:
        return PropertyCollection(_.sc_barcode_scanner_settings_get_properties(self))

    @property
    def code_direction_hint(self) -> int:
        return int(_.sc_barcode_scanner_settings_get_code_direction_hint(self))

    @code_direction_hint.setter
    def code_direction_hint(self, value: int):
        _.sc_barcode_scanner_settings_set_code_direction_hint(self, value)

    @property
    def code_duplicate_filter(self) -> int:
        return int(_.sc_barcode_scanner_settings_get_code_duplicate_filter(self))

    @code_duplicate_filter.setter
    def code_duplicate_filter(self, value: int):
        _.sc_barcode_scanner_settings_set_code_duplicate_filter(self, value)

    @property
    def max_number_of_codes_per_frame(self) -> int:
        return int(
            _.sc_barcode_scanner_settings_get_max_number_of_codes_per_frame(self)
        )

    @max_number_of_codes_per_frame.setter
    def max_number_of_codes_per_frame(self, value: int):
        _.sc_barcode_scanner_settings_set_max_number_of_codes_per_frame(self, value)

    def set_code_location_area_1d(
        self, position: Tuple[float, float], size: Tuple[float, float]
    ):
        location = _ScRectangleF.from_position_and_size(position, size)
        _.sc_barcode_scanner_settings_set_code_location_area_1d(self, location)

    @property
    def search_area(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return _.sc_barcode_scanner_settings_get_search_area(self).toList()

    @search_area.setter
    def search_area(self, value: Tuple[Tuple[float, float], Tuple[float, float]]):
        location = _ScRectangleF.from_position_and_size(value[0], value[1])
        _.sc_barcode_scanner_settings_set_search_area(self, location)

    @property
    def code_location_area_1d(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return _.sc_barcode_scanner_settings_get_code_location_area_1d(self).toList()

    @code_location_area_1d.setter
    def code_location_area_1d(
        self, value: Tuple[Tuple[float, float], Tuple[float, float]]
    ):
        location = _ScRectangleF.from_position_and_size(value[0], value[1])
        _.sc_barcode_scanner_settings_set_code_location_area_1d(self, location)

    @property
    def code_location_area_2d(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return _.sc_barcode_scanner_settings_get_code_location_area_2d(self).toList()

    @code_location_area_2d.setter
    def code_location_area_2d(
        self, value: Tuple[Tuple[float, float], Tuple[float, float]]
    ):
        location = _ScRectangleF.from_position_and_size(value[0], value[1])
        _.sc_barcode_scanner_settings_set_code_location_area_2d(self, location)

    @property
    def code_location_constraint_1d(self) -> ct.c_int:
        return _.sc_barcode_scanner_settings_get_code_location_constraint_1d(self)

    @code_location_constraint_1d.setter
    def code_location_constraint_1d(self, constraint: ct.c_int):
        _.sc_barcode_scanner_settings_set_code_location_constraint_1d(self, constraint)

    @property
    def code_location_constraint_2d(self) -> ct.c_int:
        return _.sc_barcode_scanner_settings_get_code_location_constraint_2d(self)

    @code_location_constraint_2d.setter
    def code_location_constraint_2d(self, constraint: ct.c_int):
        _.sc_barcode_scanner_settings_set_code_location_constraint_2d(self, constraint)

    @property
    def focus_mode(self) -> ct.c_int:
        return _.sc_barcode_scanner_settings_get_focus_mode(self)

    @focus_mode.setter
    def focus_mode(self, focus_mode: ct.c_int):
        _.sc_barcode_scanner_settings_set_focus_mode(self, focus_mode)


class FrameSequence(object):
    """A sequence of frames."""

    def __init__(self, context: "RecognitionContext"):
        self._context = context

    def process_frame(
        self, image_description: ct.c_void_p, data: ct.c_void_p
    ) -> ProcessFrameResult:
        return _.sc_recognition_context_process_frame(
            self._context, image_description, data
        )

    def end(self):
        _.sc_recognition_context_end_frame_sequence(self._context)
        self._context = None


class RecognitionContext(NativeHandle):
    def __init__(
        self, app_key: str, writable_data_path: str, device_name: Optional[str] = None
    ):
        handle = _.sc_recognition_context_new(
            ct.c_char_p(app_key.encode("utf-8")),
            ct.c_char_p(writable_data_path.encode("utf-8")),
            (
                ct.c_char_p(device_name.encode("utf-8"))
                if device_name is not None
                else None
            ),
        )
        NativeHandle.__init__(self, handle, _.sc_recognition_context_release)

    def start_new_frame_sequence(self) -> FrameSequence:
        _.sc_recognition_context_start_new_frame_sequence(self)
        return FrameSequence(self)

    def set_camera_properties(
        self,
        facing_direction: FacingDirection = FacingDirection.UNKNOWN,
        camera_identifier: str = "",
    ):
        _.sc_recognition_context_set_camera_properties(
            self,
            _CameraProperties(
                facing_direction.value,
                ct.c_char_p(camera_identifier.lower().strip().encode("utf-8")),
            ),
        )


class ImageDescription(NativeHandle):
    def __init__(self, handle: ct.c_void_p = None):
        if handle is None:
            handle = _.sc_image_description_new()
        NativeHandle.__init__(self, handle, _.sc_image_description_release)

    @property
    def width(self) -> int:
        return int(_.sc_image_description_get_width(self))

    @width.setter
    def width(self, value: int):
        _.sc_image_description_set_width(self, ct.c_uint32(value))

    @property
    def height(self) -> int:
        return int(_.sc_image_description_get_height(self))

    @height.setter
    def height(self, value: int):
        _.sc_image_description_set_height(self, ct.c_uint32(value))

    @property
    def layout(self) -> int:
        return int(_.sc_image_description_get_layout(self))

    @layout.setter
    def layout(self, value: int):
        _.sc_image_description_set_layout(self, value)

    @property
    def planes_count(self) -> int:
        return int(_.sc_image_description_get_planes_count(self))

    def get_plane_offset(self, plane_index: int) -> int:
        return int(
            _.sc_image_description_get_plane_offset(self, ct.c_uint32(plane_index))
        )

    def set_plane_offset(self, plane_index: int, value: int):
        _.sc_image_description_set_plane_offset(
            self, ct.c_uint32(plane_index), ct.c_ssize_t(value)
        )

    def get_plane_row_bytes(self, plane_index: int) -> int:
        return int(
            _.sc_image_description_get_plane_row_bytes(self, ct.c_uint32(plane_index))
        )

    def set_plane_row_bytes(self, plane_index: int, value: int):
        _.sc_image_description_set_plane_row_bytes(
            self, ct.c_uint32(plane_index), ct.c_uint32(value)
        )


class BarcodeScanner(NativeHandle):
    def __init__(self, context: RecognitionContext, settings: BarcodeScannerSettings):
        settings_handle = settings and settings.handle or None
        handle = _.sc_barcode_scanner_new_with_settings(context.handle, settings_handle)
        NativeHandle.__init__(self, handle, _.sc_barcode_scanner_release)
        self._session = BarcodeScannerSession(_.sc_barcode_scanner_get_session(self))

    def wait_for_setup_completed(self) -> bool:
        return bool(_.sc_barcode_scanner_wait_for_setup_completed(self))

    @property
    def session(self) -> "BarcodeScannerSession":
        return self._session


class Barcode(NativeHandle):
    """The Barcode class represents a barcode, or 2D code that has been localized or
    recognized by the barcode recognition engine."""

    def __init__(self, handle: ct.c_void_p):
        _.sc_barcode_retain(handle)
        NativeHandle.__init__(self, handle, _.sc_barcode_release)

    @property
    def data(self) -> Optional[str]:
        """The data of the barcode as a UTF-8 string.

        For barcodes that have been localized but not recognized, None is returned as
        the data string.
        """
        raw_data = self.raw_data
        if raw_data:
            encoding = self.data_encoding
            result = ""
            for enc in encoding:
                address = ct.cast(raw_data.bytes, ct.c_void_p).value
                char_data = ct.string_at(address + enc.start, enc.end - enc.start)
                result += char_data.decode(enc.encoding, "ignore")
            return result
        return None

    @property
    def is_gs1_data_carrier(self) -> bool:
        return bool(_.sc_barcode_is_gs1_data_carrier(self))

    @property
    def composite_flag(self) -> int:
        return int(_.sc_barcode_get_composite_flag(self))

    @property
    def symbol_count(self) -> int:
        return int(_.sc_barcode_get_symbol_count(self))

    @property
    def raw_data(self) -> ByteArray:
        return _.sc_barcode_get_data(self)

    @property
    def data_encoding(self) -> List[EncodingRange]:
        result = []
        encodings = _.sc_barcode_get_data_encoding(self)
        for i in range(encodings.size):
            encoding = encodings.encodings[i]
            result.append(
                EncodingRange(
                    encoding.encoding.str.decode("utf-8"),
                    encoding.start,
                    encoding.end,
                )
            )
        _.sc_encoding_array_free(encodings)
        return result

    @property
    def location(self) -> Quadrilateral:
        """The location of the localized/recognized barcode."""
        return _.sc_barcode_get_location(self)

    @property
    def symbology(self) -> int:
        """The symbology of the barcode.

        For barcodes that have been localized but not recognized, SYMBOLOGY_UNKNOWN is
        returned.
        """
        return int(_.sc_barcode_get_symbology(self))

    @property
    def symbology_string(self) -> str:
        """Convenience function for directly getting the symbology of the barcode as a
        string.

        The symbology names are all lower-case.
        """
        return _.sc_symbology_to_string(self.symbology).decode("utf-8")

    def __eq__(self, other: object):
        """They are the same if symbology and value equal, used to filter out
        duplicates."""
        if not isinstance(other, Barcode):
            return NotImplemented
        return self.symbology == other.symbology and self.data == other.data

    def __hash__(self) -> int:
        return hash((self.symbology, self.data))


class BarcodeArray(NativeHandle):
    def __init__(self, handle: ct.c_void_p):
        _.sc_barcode_array_retain(handle)
        NativeHandle.__init__(self, handle, _.sc_barcode_array_release)

    def __getitem__(self, index: int) -> Barcode:
        __barcode = _.sc_barcode_array_get_item_at(self, ct.c_uint32(index))
        if not __barcode:
            raise StopIteration
        return Barcode(__barcode)

    def __len__(self) -> int:
        return int(_.sc_barcode_array_get_size(self))


class BarcodeScannerSession(NativeHandle):
    def __init__(self, handle: ct.c_void_p):
        _.sc_barcode_scanner_session_retain(handle)
        NativeHandle.__init__(self, handle, _.sc_barcode_scanner_session_release)

    @property
    def newly_recognized_codes(self) -> BarcodeArray:
        return BarcodeArray(
            _.sc_barcode_scanner_session_get_newly_recognized_codes(self)
        )

    @property
    def newly_localized_codes(self) -> BarcodeArray:
        return BarcodeArray(
            _.sc_barcode_scanner_session_get_newly_localized_codes(self)
        )

    @property
    def all_recognized_codes(self) -> BarcodeArray:
        return BarcodeArray(_.sc_barcode_scanner_session_get_all_recognized_codes(self))

    def clear(self):
        _.sc_barcode_scanner_session_clear(self)


class ImageBuffer(NativeHandle):
    def __init__(self, _image_buffer: _ImageBuffer):
        self.__image_data = _image_buffer.contents.data
        self.__image_data_size = _image_buffer.contents.data_size
        NativeHandle.__init__(
            self, ct.addressof(_image_buffer.contents), _.sc_image_buffer_free
        )
        self.__description = ImageDescription(_image_buffer.contents.description)
        _image_buffer.contents.description = 0

    @property
    def description(self) -> ImageDescription:
        return self.__description

    @property
    def data(self) -> ct.pointer:
        return ct.cast(self.__image_data, ct.POINTER(ct.c_uint8))

    @property
    def data_size(self) -> int:
        return self.__image_data_size

    @property
    def numpy_array(self) -> np.ndarray:
        return np.ctypeslib.as_array(self.data, shape=(self.data_size,))


class BarcodeGenerator(NativeHandle):
    def __handle_error(self, error: _Error):
        message = error.message.decode("utf-8")
        _.sc_error_free(error)
        raise RuntimeError(message)

    def __init__(self, context: RecognitionContext, symbology: int):
        error = _Error()
        handle = _.sc_barcode_generator_new(context.handle, ct.c_int(symbology), error)
        if not handle:
            self.__handle_error(error)
        NativeHandle.__init__(self, handle, _.sc_barcode_generator_free)

    def set_options(self, options: Dict[str, Any]):
        error = _Error()
        assert isinstance(options, dict)
        _.sc_barcode_generator_set_options(
            self, json.dumps(options).encode("utf-8"), error
        )
        if error.message:
            self.__handle_error(error)

    def __generate_from_data(
        self, data: bytes, data_length: int, encoding_array: _EncodingRangeArray
    ) -> _ImageBuffer:
        error = _Error()
        image_buffer_ptr = _.sc_barcode_generator_generate(
            self, data, ct.c_size_t(data_length), encoding_array, error
        )
        if not image_buffer_ptr:
            self.__handle_error(error)
        return image_buffer_ptr

    def generate(self, data_str: str) -> ImageBuffer:
        try:
            data = data_str.encode("ascii")
        except UnicodeEncodeError:
            raise RuntimeError("The barcode generator only supports ASCII input.")
        data_length = len(data)
        encodings = _.sc_encoding_array_new(1)
        ascii_str = "US-ASCII"
        encodings.encodings[0].encoding.str = ascii_str.encode("utf-8")
        encodings.encodings[0].encoding.size = len(ascii_str)
        encodings.encodings[0].start = 0
        encodings.encodings[0].start = data_length
        return ImageBuffer(self.__generate_from_data(data, data_length, encodings))


class TrackedObject(NativeHandle):
    """
    An object that is being tracked by the object tracking module.
    """

    def __init__(self, handle: ct.c_void_p):
        _.sc_tracked_object_retain(handle)
        NativeHandle.__init__(self, handle, _.sc_tracked_object_release)

    @property
    def location(self) -> Quadrilateral:
        """
        :return: The current position of the tracked object
        """
        return _.sc_tracked_object_get_location(self)

    @property
    def id(self) -> int:
        """
        :return: A unique identifier for the tracked object. If the tracker loses the object,
                 it might assign a new ID.
        """
        return _.sc_tracked_object_get_id(self)

    def get_barcode(self) -> Optional[Barcode]:
        """
        :return: The tracked barcode if the tracked object is a barcode, else None
        """
        __barcode = _.sc_tracked_object_get_barcode(self)
        if __barcode is None:
            return None
        return Barcode(__barcode)


class TrackedObjectMap(NativeHandle):
    def __init__(self, handle: ct.c_void_p):
        NativeHandle.__init__(self, handle, _.sc_tracked_object_map_release)

    def __getitem__(self, key: int) -> TrackedObject:
        __object = _.sc_tracked_object_map_get_item_at(self, ct.c_uint(key))
        if __object is None:
            raise KeyError(f"Object with {key} not found")
        return TrackedObject(__object)

    def __contains__(self, key: int) :
        return _.sc_tracked_object_map_get_item_at(self, ct.c_uint(key)) is not None

    def __len__(self) :
        return int(_.sc_tracked_object_map_get_size(self))

    def items(self) :
        return [(key, self.__getitem__(key)) for key in self.keys()]

    def keys(self) :
        num_keys = self.__len__()
        keys_ptr = _.sc_tracked_object_map_get_ids(self)
        return keys_ptr[:num_keys]


class ObjectTrackerSession:
    """
    The object tracking session allows to query the currently tracked objects
    """

    _handle: ct.c_void_p

    def __init__(self, handle: ct.c_void_p):
        self._handle = handle

    @property
    def tracked_objects(self) -> TrackedObjectMap:
        """
        :return: The currently tracked objects, grouped by their unique identifier
        """
        return TrackedObjectMap(
            _.sc_object_tracker_session_get_tracked_objects(self._handle)
        )


class ObjectTracker(NativeHandle):
    """
    The object tracker allows to detect and track one or multiple barcodes.
    """

    _session: ObjectTrackerSession

    def __init__(
        self, context: RecognitionContext, scanner_settings: BarcodeScannerSettings
    ):
        NativeHandle.__init__(
            self, _.sc_object_tracker_new(context.handle), _.sc_object_tracker_release
        )
        _.sc_object_tracker_apply_scanner_settings(self, scanner_settings.handle)
        self._session = ObjectTrackerSession(_.sc_object_tracker_get_session(self))

    @property
    def session(self) -> ObjectTrackerSession:
        """
        :return: The object tracking session
        """
        return self._session
