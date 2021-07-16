class Config:
    def __init__(self, **kwargs):
        self.defaults()
        for k, v in kwargs.items():
            setattr(self, k, v)

    def defaults(self):
        self.ROOT = 'resource/maps/procedural/map'
        self.SUFFIX = '/map.tmx'
        self.SZ = 7
        self.BORDER = 1
        self.SZH = self.SZ
        self.SZV = self.SZ
        self.R = self.SZH + self.BORDER
        self.C = self.SZV + self.BORDER

        self.STIM = 7
        self.NENT = 2

        # Base agent stats
        self.HEALTH = 10
        self.FOOD = 32
        self.WATER = 32

        # Attack ranges
        self.MELEERANGE = 1
        self.RANGERANGE = 0
        self.MAGERANGE = 0

        self.FOREST_LOCATIONS = [(6, 3), (6, 5)]

    def SPAWN(self):
        return self.SZV // 2 + self.BORDER, self.SZH // 2 + self.BORDER

    # Damage formulas. Lambdas don't pickle well
    def MELEEDAMAGE(self, ent, targ):
        return 0

    def RANGEDAMAGE(self, ent, targ):
        return 0

    def MAGEDAMAGE(self, ent, targ):
        return 0
