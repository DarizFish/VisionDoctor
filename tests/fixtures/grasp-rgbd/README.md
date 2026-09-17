# Real RGB-D capture fixture

Captured from the PICK-A17 Gazebo RGB-D camera on 2026-09-12 after restarting the
stopped simulation server. Original capture:
`.runtime/gazebo-pick-cell/captures/capture-20260912T081410Z-4b042919`.

RGB and metadata are unchanged. `depth.npz` losslessly compresses `depth.npy`;
tests restore it before executing the public demonstration perception code.
The two physical scene centers, used only by the test, are A=(0.5,-0.1,0.1) and
B=(0.5,-0.3,0.1) meters. The detector does not read these reference positions.
It estimates position and uses the fixture's declared orientation.
