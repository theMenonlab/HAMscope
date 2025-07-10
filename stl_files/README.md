# HAMscope 3D Printable Components

This directory contains STL files for 3D printing custom hardware components designed specifically for the HAMscope hyperspectral microscopy system. These components enable precise control of illumination, filtering, and sample positioning.

#### `20250221_mirror_spinner_round.stl`
A shortpass dichroic mirror (400 nm cutoff; Newport, 10SWF-400-B) positioned at the Fourier plane directs light toward the reference path. To maximize optical throughput during miniscope image capture, the dichroic is mounted on a motorized rotation stage and can be retracted from the beam path. The mount is custom 3D-printed and driven by a NEMA 17 stepper motor (Stepper-online).

#### `LVBF holder.stl`
For the sturdy attachment of a linear variable bandpass filter (LVBF; Edmund Optics, 88-365) to a linear CNC stage (RATTMMOTOR, ZBX80, 100 mm stroke).

#### `plant_clamp.stl`
Simple clamp for holding a small poplar branch for imaging.


### 🔆 Supplementary LED Illumination System

#### `LED_front.stl`
Attaches to the excitation port of the miniscope with wire, delivering supplementary illumination. A collimating lens and emission filter are positioned within this optical tube.

#### `LED_back.stl`
This is the second half of the supplementary illumination system. It holds the LED and its heat sink. 

#### `LED_spacer.stl`
The spacer connects the two above parts of the bracket.

#### `LED_bracket.stl`
Additional support, holding the LED heat sink to the main optical rod.

## Printing Guidelines

**Recommended Printer**: Bambu Labs A1M with 0.2 or 0.1 mm nozzle
**Materials**: Black PLA or PETG for optimal light absorption and thermal stability
**Settings**: Standard quality settings appropriate for precision optical components