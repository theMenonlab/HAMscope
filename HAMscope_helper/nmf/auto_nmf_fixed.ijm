inputDir = "/media/al/Extreme SSD/20250425_dataset/dataset_split/split_1/train/ham_nonorm";
outputDir = "/media/al/Extreme SSD/20250425_dataset/dataset_split/split_1/train/ham_nonorm_nmf";

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
  run ( "PoissonNMF " , "number=3 _number=2 subsamples=3 segregation=0 saturation=65535 _background=0 _background_0=[Minimal values] dye_1=spectrum1_100.emn dye_2=spectrum2_100.emn dye_3=spectrum3_100.emn dye dye_0 dye_4");

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



inputDir = "/media/al/Extreme SSD/20250425_dataset/dataset_split/split_1/test/ham_nonorm";
outputDir = "/media/al/Extreme SSD/20250425_dataset/dataset_split/split_1/test/ham_nonorm_nmf";

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
  run ( "PoissonNMF " , "number=3 _number=2 subsamples=3 segregation=0 saturation=65535 _background=0 _background_0=[Minimal values] dye_1=spectrum1_100.emn dye_2=spectrum2_100.emn dye_3=spectrum3_100.emn dye dye_0 dye_4");

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




inputDir = "/media/al/Extreme SSD/20250425_dataset/dataset_split/split_2/train/ham_nonorm";
outputDir = "/media/al/Extreme SSD/20250425_dataset/dataset_split/split_2/train/ham_nonorm_nmf";

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
  run ( "PoissonNMF " , "number=3 _number=2 subsamples=3 segregation=0 saturation=65535 _background=0 _background_0=[Minimal values] dye_1=spectrum1_100.emn dye_2=spectrum2_100.emn dye_3=spectrum3_100.emn dye dye_0 dye_4");

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



inputDir = "/media/al/Extreme SSD/20250425_dataset/dataset_split/split_2/test/ham_nonorm";
outputDir = "/media/al/Extreme SSD/20250425_dataset/dataset_split/split_2/test/ham_nonorm_nmf";

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
  run ( "PoissonNMF " , "number=3 _number=2 subsamples=3 segregation=0 saturation=65535 _background=0 _background_0=[Minimal values] dye_1=spectrum1_100.emn dye_2=spectrum2_100.emn dye_3=spectrum3_100.emn dye dye_0 dye_4");

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



inputDir = "/media/al/Extreme SSD/20250425_dataset/dataset_split/split_3/train/ham_nonorm";
outputDir = "/media/al/Extreme SSD/20250425_dataset/dataset_split/split_3/train/ham_nonorm_nmf";

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
  run ( "PoissonNMF " , "number=3 _number=2 subsamples=3 segregation=0 saturation=65535 _background=0 _background_0=[Minimal values] dye_1=spectrum1_100.emn dye_2=spectrum2_100.emn dye_3=spectrum3_100.emn dye dye_0 dye_4");

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



inputDir = "/media/al/Extreme SSD/20250425_dataset/dataset_split/split_3/test/ham_nonorm";
outputDir = "/media/al/Extreme SSD/20250425_dataset/dataset_split/split_3/test/ham_nonorm_nmf";

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
  run ( "PoissonNMF " , "number=3 _number=2 subsamples=3 segregation=0 saturation=65535 _background=0 _background_0=[Minimal values] dye_1=spectrum1_100.emn dye_2=spectrum2_100.emn dye_3=spectrum3_100.emn dye dye_0 dye_4");

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



inputDir = "/media/al/Extreme SSD/20250425_dataset/dataset_split/split_4/test/ham_nonorm";
outputDir = "/media/al/Extreme SSD/20250425_dataset/dataset_split/split_4/test/ham_nonorm_nmf";

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
  run ( "PoissonNMF " , "number=3 _number=2 subsamples=3 segregation=0 saturation=65535 _background=0 _background_0=[Minimal values] dye_1=spectrum1_100.emn dye_2=spectrum2_100.emn dye_3=spectrum3_100.emn dye dye_0 dye_4");

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


inputDir = "/media/al/Extreme SSD/20250425_dataset/dataset_split/split_4/train/ham_nonorm";
outputDir = "/media/al/Extreme SSD/20250425_dataset/dataset_split/split_4/train/ham_nonorm_nmf";

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
  run ( "PoissonNMF " , "number=3 _number=2 subsamples=3 segregation=0 saturation=65535 _background=0 _background_0=[Minimal values] dye_1=spectrum1_100.emn dye_2=spectrum2_100.emn dye_3=spectrum3_100.emn dye dye_0 dye_4");

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