<h1>Vehicle Speed Detection </h1>
<ul>
<li>This is a python script to detect speed of multiple vehicles on multi-lane highways.</li>
<li>It uses Haar Cascade Classifier to detect vehicles in the every nth frame.</li>
<li>It removes unnecessary portion from the image to speed up processing.</li>
<li>Two reference lines have been set, one for vehicle entry and one for exit.</li>
<li>When any vehicle in any lane crosses the entry point, the time is recorded, and the vehicle is tracked.</li>
<li>Tracking is done using centroid tracking Techniques.</li>
<li>Time is recorded when the vehicles crosses the exit line.</li>
<li>Based on the time difference, the vehicles speed is estimated.</li>
</ul>

<h4>Note : A detailed version of this project also exists. Check out my other repositories.</h4>
