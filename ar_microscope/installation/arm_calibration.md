## Calibrating the microdisplay of the ARM

The ARM's microdisplay can be slightly misaligned with the sensor's original
image. To calibrate the microdisplay, the user will need a target slide, such as
[this](https://www.thorlabs.com/thorproduct.cfm?partnumber=R1L1S1N), that has a
concentric circle pattern. Here are the steps for the calibration:

1. Roughly center the target slide's concentric circles under the 40x objective.
Note that the auto-brightness adjustment may not work well with the slide, and
the brightness slider may have to be turned down to 0 before being re-adjusted
to a low value.

2. Run the ARM in calibration mode via command-line:
`bash /usr/local/bin/ar_microscope --calibration_mode=1`

3. Precisely center the target slide so that the concentric circles align with
those shown in the ARM software.

4. Use the left and top margin spinboxes to center the microdisplay's concentric
circles with those of the target slide.

5. Note the numbers in the left and margin spinboxes. Edit the ARM config (at
`/usr/local/share/arm_config.textproto`) by adding these numbers to the
corresponding margin numbers. Note that the config will need to be edited under
sudo.
