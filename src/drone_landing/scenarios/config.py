from dataclasses import dataclass, fields


@dataclass
class ScenarioConfig:
    """Bounds and ranges the generator samples a platform trajectory from."""

    bounds_x: float = 8.0
    bounds_y: float = 8.0
    min_waypoints: int = 3
    max_waypoints: int = 8
    min_speed: float = 0.5
    max_speed: float = 2.0
    min_altitude: float = 0.5
    max_altitude: float = 0.5
    min_spacing: float = 2.0
    start: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def default(cls, field_name: str):
        """The declared default for one field, so CLI flags need not restate them."""
        for field in fields(cls):
            if field.name == field_name:
                return field.default
        raise KeyError(field_name)

    def to_dict(self) -> dict:
        return {
            "bounds_x": self.bounds_x,
            "bounds_y": self.bounds_y,
            "min_waypoints": self.min_waypoints,
            "max_waypoints": self.max_waypoints,
            "min_speed": self.min_speed,
            "max_speed": self.max_speed,
            "min_altitude": self.min_altitude,
            "max_altitude": self.max_altitude,
            "min_spacing": self.min_spacing,
            "start": list(self.start),
        }
