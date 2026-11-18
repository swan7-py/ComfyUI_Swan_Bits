class SWAN_SAM3DTrackPalette:
    """Optional: collect one colour per tracked person from external pickers.

    Both Swan SAM3DBody Solid Colors (video) and Swan SAM3D Track Colors (GLB)
    take a comma separated hex list and fall back to an automatic even-hue spread
    when it is empty, so this node is entirely optional. Leave it unwired and any
    number of people gets a distinct colour for free.

    Colours arrive as linkable STRING inputs rather than node widgets: slots are
    added by count, and a linked slot stores its link by name, so growing the list
    never shifts earlier values. Wire LayerUtility: ColorPicker (or any string
    node) into each slot.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inputcount": ("INT", {
                    "default": 2, "min": 1, "max": 64, "step": 1,
                    "tooltip": "How many colour inputs to expose. "
                               "Click 'Update inputs' after changing this.",
                }),
                "color_1": ("STRING", {
                    "default": "#ffffff",
                    "forceInput": True,
                    "tooltip": "Colour for person 1, in SAM3 track order.",
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("colors",)
    FUNCTION = "build"
    CATEGORY = "SwanBits/3D"
    DESCRIPTION = (
        "Collects hex colours from external picker nodes into the list both Swan "
        "SAM3D colour nodes consume. Unwired = automatic colours."
    )

    def build(self, inputcount, **colors):
        picked = [colors.get("color_%d" % i, "") for i in range(1, int(inputcount) + 1)]
        return (",".join(c for c in picked if c),)
