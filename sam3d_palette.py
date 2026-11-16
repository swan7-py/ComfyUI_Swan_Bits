class SWAN_SAM3DTrackPalette:
    """Pick one colour per tracked person with the UI colour picker.

    Both Swan SAM3DBody Solid Colors (video) and Swan SAM3D Track Colors (GLB)
    take a comma separated hex list. Driving them from here means the palette
    only has to be edited once and the video and the model can't drift apart.
    Disconnect the outputs (or bypass this node) and both renderers fall back to
    their automatic hue spread.

    Eight slots: the first four keep the original even-hue quarter spread so
    existing results don't shift, the last four are the half-steps between them.
    """

    @classmethod
    def INPUT_TYPES(cls):
        defaults = {
            "color_1": "#e65050",
            "color_2": "#9be650",
            "color_3": "#50e6e6",
            "color_4": "#9b50e6",
            "color_5": "#e69950",
            "color_6": "#50e699",
            "color_7": "#5099e6",
            "color_8": "#e65099",
        }
        slots = {}
        for name, default in defaults.items():
            slots[name] = ("COLOR", {
                "default": default,
                "tooltip": "Person %s, counting in the order SAM3 tracks them. "
                           "Slots beyond the tracked count are ignored."
                           % name.rsplit("_", 1)[1],
            })
        return {"required": slots}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("colors",)
    FUNCTION = "build"
    CATEGORY = "SwanBits/3D"
    DESCRIPTION = (
        "Colour pickers for up to eight tracked people. Wire the hex list into "
        "both Swan SAM3D colour nodes so the video and the exported GLB match."
    )

    def build(self, color_1, color_2, color_3, color_4,
              color_5, color_6, color_7, color_8):
        return (",".join([color_1, color_2, color_3, color_4,
                          color_5, color_6, color_7, color_8]),)
