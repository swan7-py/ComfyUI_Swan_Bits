SLOT_COUNT = 12


class SWAN_SAM3DTrackPalette:
    """Optional: pin one colour per tracked person.

    Both Swan SAM3DBody Solid Colors (video) and Swan SAM3D Track Colors (GLB)
    take a comma separated hex list and fall back to an automatic even-hue spread
    when it is empty, so this node is entirely optional. Leave it unwired and any
    number of people gets a distinct colour for free; wire it in only when you
    want to dictate the colours yourself.

    Twelve plain COLOR widgets rather than one list widget: the frontend's COLORS
    control is registered with hideInPanel, so it only shows up in the Node 2.0
    sidebar and is invisible on the node in the classic UI.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "color_%d" % i: ("COLOR", {
                    "default": "#ffffff",
                    "tooltip": "Person %d, counting in SAM3 track order. "
                               "Slots past the tracked count are ignored." % i,
                })
                for i in range(1, SLOT_COUNT + 1)
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("colors",)
    FUNCTION = "build"
    CATEGORY = "SwanBits/3D"
    DESCRIPTION = (
        "Optional colour overrides for up to %d tracked people. Wire the hex list "
        "into both Swan SAM3D colour nodes; leave unwired for automatic colours."
        % SLOT_COUNT
    )

    def build(self, **colors):
        picked = [colors["color_%d" % i] for i in range(1, SLOT_COUNT + 1)]
        return (",".join(c for c in picked if c),)
