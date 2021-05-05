# hot_tap_challenge
Code and report for an axisymmetric model to simulate hot tap welding process

Code files:
linear_model.py
nonlinear_model. py

The animations of the simulation results are shown in 
linear_model.mp4
nonlinear_model.py

Created with Python 3.8.5
with the following libraries
fenics 2019.1.0 (with mshr)
numpy 1.17.4
matplotlib 3.1.2

To Run the linear heat transfer model, run linear_model.py
"python3 linear_model.py"

This will create a folder "Results" with paraview files and .png files with the results
The image files can be used to create animations using 
ffmpeg package with the command :

"ffmpeg -r 10 -f image2 -s 1920x1080 -i linear_model%04d.png -vcodec libx264 -crf 10 -pix_fmt yuv420p linear_model.mp4"

or imagemagick package with the command
"convert -delay 10 linear_model*_ms.png linear_model.gif"

