inputDir = "/media/al/Extreme SSD/20250425_dataset/example_depth_stitch/input_hyperspectral_norm";
outputDir = "/media/al/Extreme SSD/20250425_dataset/example_depth_stitch/nmf_norm_4ch_roi2";

// Create output directory if it doesn't exist
File.makeDirectory(outputDir);

// Get a list of all directories starting with "set_"
list = getFileList(inputDir);
// Loop through each item
for (i = 0; i < list.length; i++) {
  // Skip any items ending in ".txt"
  if (endsWith(list[i], ".txt")) {
    continue;
  }
  
  // Process remaining items
  open(inputDir + "/" + list[i]);
  //run("PoissonNMF ", "number=4 _number=2 subsamples=4 segregation=0 saturation=65535 _background=0 _background_0=[Minimal values] dye_1=spectrum_chlorophyll.emn dye_2=spectrum_other.emn dye_3=spectrum_g_lignin.emn dye_4=spectrum_s_lignin.emn dye dye_0 dye_5 dye_6");
  run("PoissonNMF ", "number=4 _number=2 subsamples=4 segregation=0 saturation=65535 _background=0 _background_0=[Minimal values] dye_1=spectrum1_roi2.emn dye_2=spectrum2_roi2.emn dye_3=spectrum3_roi2.emn dye_4=spectrum4_roi2.emn dye dye_0 dye_5 dye_6");

  //wait(2000); // Wait for 30 seconds (30,000 milliseconds)
  
  selectImage("NMF sources");
  stack = list[i];
  //stack = list[i].substring(0, list[i].lastIndexOf("/"));
  //outputFilename = outputDir + "/" + stack + "_nmf.tif";
  outputFilename = outputDir + "/" + stack.replace(".tif", "") + "_nmf.tif";
  saveAs("TIF", outputFilename);
  
  selectImage(stack.replace(".tif", "") + "_nmf.tif");
  close();
  if (isOpen("NMF sources")) {
    close("NMF sources");}
  if (isOpen("PoissonNMF spectra")) {
    close("PoissonNMF spectra");}

  selectImage(stack);
  close();
  
}