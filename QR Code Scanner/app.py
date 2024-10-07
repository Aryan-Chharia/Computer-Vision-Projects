from flask import Flask, send_file, render_template
import os

app = Flask(__name__)

# view results.
@app.route('/view-result', methods=['GET'])
def view_scanned_data():
    if os.path.exists('result.txt'):
        with open('result.txt', 'r') as file:
            scanned_data = file.read()
        return render_template('view-result', scanned_data=scanned_data)
    else:
        return "<h1>Scanned Data</h1><p>No data available.</p>"

# download results.
@app.route('/download-result', methods=['GET'])
def download_scanned_data():
    if os.path.exists('result.txt'):
        return send_file('result.txt', as_attachment=True)
    else:
        return "<p>No data available for download.</p>"

if __name__ == "__main__":
    app.run(debug=True)
