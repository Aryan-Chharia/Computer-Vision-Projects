from flask import Flask, send_file, render_template, abort
import os

app = Flask(__name__)

# View results.
@app.route('/view-result', methods=['GET'])
def view_scanned_data():
    """Displays the scanned data from the result.txt file."""
    if os.path.exists('result.txt'):
        with open('result.txt', 'r') as file:
            scanned_data = file.read()
        return render_template('view-result.html', scanned_data=scanned_data)  # Fixed the template file extension
    else:
        return "<h1>Scanned Data</h1><p>No data available.</p>"

# Download results.
@app.route('/download-result', methods=['GET'])
def download_scanned_data():
    """Downloads the scanned data from the result.txt file."""
    file_path = 'result.txt'
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return "<p>No data available for download.</p>"

if __name__ == "__main__":
    app.run(debug=True)
