document.querySelector("form").onsubmit = function(event) {
    event.preventDefault();
    let file = document.querySelector("input[type=file]").files[0];

    if (file) {
        let reader = new FileReader();
        reader.onload = function(e) {
            let arrayBuffer = e.target.result;
            let binaryString = String.fromCharCode.apply(null, new Uint8Array(arrayBuffer));
            let workbook = XLSX.read(binaryString, { type: 'binary' });
            let sheetName = workbook.SheetNames[0];
            let sheet = workbook.Sheets[sheetName];
            let html = XLSX.utils.sheet_to_html(sheet);
            document.getElementById("excel-viewer").innerHTML = html;
        };
        reader.readAsArrayBuffer(file);
    }
};

function fetchStats() {
    fetch('/statistics')
    .then(response => response.json())
    .then(data => {
        document.getElementById('precision').textContent = data.Precision;
        document.getElementById('recall').textContent = data.Recall;
        document.getElementById('auprc').textContent = data.AUPRC;
    });
}

fetchStats();
