from dataclasses import dataclass

@dataclass
class Start:
    pass

@dataclass
class Stop:
    pass

@dataclass
class Preview:
    """Run the pipeline in preview mode (imshow, no alerts)."""
    pass
