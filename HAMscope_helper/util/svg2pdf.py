import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

def convert_svg_to_pdf():
    """Convert SVG to PDF with proper size scaling using rsvg-convert."""
    
    # Hard-coded paths
    input_path = "/home/al/Documents/hyperspectral_miniscope_paper/suberin_fig_dc.svg"
    output_path = "/home/al/Documents/hyperspectral_miniscope_paper/"
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Determine if input is file or directory and collect SVG files
    if input_path.is_file() and input_path.suffix == '.svg':
        svg_files = [input_path]
        output_dir = output_path if output_path.is_dir() else output_path.parent
    elif input_path.is_dir():
        svg_files = list(input_path.glob("*.svg"))
        output_dir = output_path
    else:
        print(f"Error: {input_path} is not a valid SVG file or directory")
        return
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not svg_files:
        print(f"No SVG files found in {input_path}")
        return
    
    print(f"Found {len(svg_files)} SVG file(s) to convert...")
    
    for svg_file in svg_files:
        pdf_filename = svg_file.stem + ".pdf"
        pdf_path = output_dir / pdf_filename
        
        try:
            # Method 1: Try rsvg-convert (more reliable for size control)
            print(f"Converting {svg_file.name}...")
            
            # Set maximum width to 1000mm (about 39 inches)
            max_width_mm = 1000
            
            # Convert using rsvg-convert with explicit width limit
            cmd = [
                "rsvg-convert",
                "--format=pdf",
                f"--width={max_width_mm * 3.78:.0f}",  # Convert mm to pixels (roughly 96 DPI)
                "--keep-aspect-ratio",
                f"--output={pdf_path}",
                str(svg_file)
            ]
            
            print(f"  Command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True)
            
            print(f"✓ Converted: {svg_file.name} → {pdf_filename}")
            
            # Verify output size
            try:
                verify_result = subprocess.run([
                    "pdfinfo", str(pdf_path)
                ], capture_output=True, text=True, check=True)
                
                for line in verify_result.stdout.split('\n'):
                    if 'Page size:' in line:
                        size_str = line.split('Page size:')[1].strip()
                        print(f"  Final PDF size: {size_str}")
                        
                        # Convert to mm for verification
                        if 'pts' in size_str:
                            parts = size_str.split(' x ')
                            if len(parts) == 2:
                                width_pts = float(parts[0])
                                height_pts = float(parts[1].split(' ')[0])
                                width_mm = width_pts * 0.352778
                                height_mm = height_pts * 0.352778
                                print(f"  Final size in mm: {width_mm:.0f} × {height_mm:.0f} mm")
                        break
            except:
                print("  Could not verify PDF size")
                
        except FileNotFoundError:
            print("rsvg-convert not found, trying alternative method...")
            
            # Method 2: Fallback to ImageMagick (if available)
            try:
                cmd = [
                    "convert",
                    "-density", "96",
                    "-resize", "3780x",  # Force max width in pixels
                    str(svg_file),
                    str(pdf_path)
                ]
                
                print(f"  Fallback command: {' '.join(cmd)}")
                subprocess.run(cmd, check=True, capture_output=True)
                print(f"✓ Converted with ImageMagick: {svg_file.name} → {pdf_filename}")
                
            except FileNotFoundError:
                print("ImageMagick also not found, trying Inkscape with temporary SVG...")
                
                # Method 3: Create a modified SVG with smaller viewBox
                try:
                    # Read and modify the SVG
                    tree = ET.parse(svg_file)
                    root = tree.getroot()
                    
                    # Set viewBox to reasonable size (1000mm x proportional height)
                    # Assuming original is very large, scale it down
                    root.set('width', '1000mm')
                    root.set('height', '300mm')  # Rough proportion
                    root.set('viewBox', '0 0 1000 300')
                    
                    # Save temporary SVG
                    temp_svg = svg_file.parent / f"temp_{svg_file.name}"
                    tree.write(temp_svg)
                    
                    # Convert the modified SVG
                    cmd = [
                        "inkscape",
                        "--export-type=pdf",
                        f"--export-filename={pdf_path}",
                        str(temp_svg)
                    ]
                    
                    subprocess.run(cmd, check=True, capture_output=True)
                    
                    # Clean up
                    temp_svg.unlink()
                    
                    print(f"✓ Converted with modified SVG: {svg_file.name} → {pdf_filename}")
                    
                except Exception as e:
                    print(f"✗ All methods failed: {e}")
                    
        except subprocess.CalledProcessError as e:
            print(f"✗ Conversion failed: {e}")

if __name__ == "__main__":
    convert_svg_to_pdf()
    print("\nConversion complete!")
    print("\nTo install missing tools:")
    print("  sudo apt install librsvg2-bin  # for rsvg-convert")
    print("  sudo apt install imagemagick   # for convert")