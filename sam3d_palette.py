class SWAN_SAM3DTrackPalette:
    """Pick one colour per tracked person with the UI colour picker.

    Both Swan SAM3DBody Solid Colors (video) and Swan SAM3D Track Colors (GLB)
    take a comma separated hex list. Driving them from here means the palette
    only has to be edited once and the video and the model can't drift apart.
    The defaults are the same even-hue spread those nodes fall back to, so an
    untouched palette behaves exactly like leaving `colors` empty.
    """

    @classmethod
    def INPUT_TYPES(cls):
        note = ("Person slot %d, counting in the order SAM3 tracks them. "
                "Slots beyond the tracked count are ignored.")
        return {
            "required": {
                "color_1": ("COLOR", {"default": "#e65050", "tooltip": note % 1}),
                "color_2": ("COLOR", {"default": "#9be650", "tooltip": note % 2}),
                "color_3": ("COLOR", {"default": "#50e6e6", "tooltip": note % 3}),
                "color_4": ("COLOR", {"default": "#9b50e6", "tooltip": note % 4}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("colors",)
    FUNCTION = "build"
    CATEGORY = "SwanBits/3D"
    DESCRIPTION = (
        "Colour pickers for up to four tracked people. Wire the hex list into "
        "both Swan SAM3D colour nodes so the video and the exported GLB match."
    )

    def build(self, color_1, color_2, color_3, color_4):
        return (",".join([color_1, color_2, color_3, color_4]),)
