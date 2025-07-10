//inputDir = "/media/al/Extreme SSD/20250425_results/results/results_30/20250425_0gan_single_hs/test_latest/images";
//outputDir = "/media/al/Extreme SSD/20250425_results/results/results_30/20250425_0gan_single_hs/test_latest/images_nmf_3ch";
inputDir = "/media/al/Extreme SSD/20250425_results/results/new_layer_norm/results/20250425_0gan_single_hs_1/test_latest/images"
outputDir = "/media/al/Extreme SSD/20250425_results/results/new_layer_norm/results/20250425_0gan_single_hs_1/test_latest/images_nmf"


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
  
  // convert to 16 bit
  setOption("ScaleConversions", true);
  run("16-bit");
  
    // Process remaining items
  open(inputDir + "/" + list[i]);
  
  // Check if stack has more than 30 slices and reduce if necessary
  numSlices = nSlices;
  if (numSlices > 30) {
    run("Make Substack...", "slices=1-30");
    // Close the original stack and work with the substack
    selectImage(list[i]);
    close();
    // The substack becomes the active image
  }
  
  setOption("ScaleConversions", true);
  run("16-bit");
  
  // run poissonNMF
  run("PoissonNMF ", "number=3 _number=2 subsamples=3 segregation=0 saturation=65535 _background=0 _background_0=[Minimal values] dye_1=spectrum1_100.emn dye_2=spectrum2_100.emn dye_3=spectrum3_100.emn dye dye_0 dye_4");
  
  selectImage("NMF sources");
  stack = list[i];
  //stack = list[i].substring(0, list[i].lastIndexOf("/"));
  //outputFilename = outputDir + "/" + stack + "_nmf.tif";
  outputFilename = outputDir + "/" + stack.replace(".tif", "") + ".tif";
  saveAs("TIF", outputFilename);
  
  selectImage(stack.replace(".tif", "") + ".tif");
  close();
  if (isOpen("NMF sources")) {
    close("NMF sources");}
  if (isOpen("PoissonNMF spectra")) {
    close("PoissonNMF spectra");}

  //selectImage("Substack (1-30)");
  close("*");
}