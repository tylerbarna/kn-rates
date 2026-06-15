import survey_sim
import survey_sim.survey_sim as core

print("survey_sim package:", survey_sim.__file__)
print("core module:", core.__file__)
print("Has DetectedSource:", hasattr(core, "DetectedSource"))
print([n for n in dir(core) if "Detected" in n or "Source" in n])