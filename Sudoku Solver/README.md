# Sudoku Solver with Computer Vision

## Sudoku Solver Project Overview

This project utilizes computer vision techniques to solve Sudoku puzzles from natural images. The process involves capturing an image of a Sudoku puzzle, extracting the puzzle grid, identifying and classifying digits within each cell, solving the puzzle using a recursive backtracking algorithm, and finally overlaying the solution onto the original image. OpenCV is employed for image processing tasks.

## Implementation Details

Image processing begins with applying an adaptive threshold to obtain a binary image. Contours are then computed from the binary image to locate the main puzzle grid. A perspective transform is applied to achieve a bird's eye view of the puzzle grid. Subsequently, contours within the transformed grid are computed to identify individual cells. We ascertain which cells contain digits and store relevant information about each cell to reconstruct the Sudoku grid. Upon solving the puzzle, the solution numbers are integrated into the grid cells, and the inverse perspective transform is applied to place the solution back onto the original image.

For the purpose of solving puzzles, a two-dimensional array representing the puzzle grid is provided to an instance of the `SudokuSolver` class. A recursive backtracking algorithm is employed to solve the sudoku puzzle. It's important to note that the solver only returns a single solution, even in cases where multiple solutions may exist.

## Tech Stack

- Python 3.x
- TensorFlow 2.x
- OpenCV 4.x
- Imutils
- NumPy

## Installation
Follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Vatsal565/Sudoku-Solver.git
   cd Sudoku-Solver
   ```
2. Set up a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
3. Install the required dependencies:
   ``` bash
   pip install -r requirements.txt
   ```

### Running the Tool


1. Place your sudoku puzzle images in the `sudoku` directory.
2. Update the `IMG` variable in the `main.py` file with the path of the image you want to solve.
3. Update the `MODEL` variable in the `main.py` file with the path to the pre-trained model.
4. Start the application:
   ```bash
   python main.py
   ```

5. Wait for the cv2 to display original image and then press any key to continue.
6. The solved sudoku puzzle will be displayed in a new window.


## Contact

For questions or feedback:
- 📧 Email: vatsalbateriwala562005@gmail.com
- 🐙 GitHub: [@Vatsal565](https://github.com/Vatsal565)

---

<div align="center">

**Made with ❤️ by Vatsal Bateriwala**

</div>