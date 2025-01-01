// Toggle camera state
function toggleCamera() {
    fetch('/toggle_camera', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            const button = document.getElementById('toggle-camera');
            button.textContent = data.camera_active ? 'Turn Off Camera' : 'Turn On Camera';
        });
}

// Toggle visualization overlay
function toggleVisualization() {
    fetch('/toggle_visualization', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            alert('Visualization is now ' + (data.display_visualization ? 'ON' : 'OFF'));
        })
        .catch(error => console.error('Error toggling visualization:', error));
}

function generateGradCam() {
    fetch('/generate_grad_cam', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
            } else {
                alert('Grad-CAM generated successfully!');
                const gradCamImage = document.getElementById('grad-cam-stream');
                gradCamImage.style.display = 'block'; // 顯示圖片
                gradCamImage.src = '/grad_cam_feed?timestamp=' + new Date().getTime(); // 刷新圖片
            }
        })
        .catch(error => console.error('Error generating Grad-CAM:', error));
}

// Refresh Grad-CAM image
function refreshGradCam() {
    const gradCamImage = document.getElementById('grad-cam-stream');
    gradCamImage.src = '/grad_cam_feed?timestamp=' + new Date().getTime();
}

// Update probabilities
function updateProbabilities() {
    fetch('/get_probabilities')
        .then(response => response.json())
        .then(data => {
            if (data.probabilities) {
                const tableBody = document.getElementById('probabilities-table').querySelector('tbody');
                tableBody.innerHTML = ''; // 清空表格內容

                // 提取所有概率值，計算最大值和最小值
                const probabilities = data.probabilities.map(item => item.probability);
                const maxProbability = Math.max(...probabilities);
                const minProbability = Math.min(...probabilities);

                // 遍歷每個數據項，計算顏色並填充表格
                data.probabilities.forEach(item => {
                    const normalizedValue = (item.probability - minProbability) / (maxProbability - minProbability);
                    const colorIntensity = Math.round(255 * (1 - normalizedValue)); // 值越大顏色越深
                    const backgroundColor = `rgb(${colorIntensity}, ${colorIntensity}, 255)`;

                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td style="padding: 5px; color: black;">${item.label}</td>
                        <td style="padding: 5px; background-color: ${backgroundColor}; color: black;">${item.probability.toFixed(2)}%</td>
                    `;
                    tableBody.appendChild(row);
                });
            }
        })
        .catch(error => console.error('Error fetching probabilities:', error));
}

let currentModel = "alexnet_gnn";

function switchModel(model) {
    fetch('/switch_model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: model })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert(data.message); // 显示模型切换成功的消息
            currentModel = model; // 更新当前模型
            updateCurrentModelDisplay(); // 更新显示
        } else {
            alert(data.message); // 显示错误消息
        }
    })
    .catch(error => console.error('Error switching model:', error));
}

function updateCurrentModelDisplay() {
    const modelDisplay = document.getElementById('current-model-name');
    if (currentModel === 'alexnet_gnn') {
        modelDisplay.textContent = "AlexNet + GNN";
    } else if (currentModel === 'vgg19_gnn') {
        modelDisplay.textContent = "VGG19 + GNN";
    }
}

const categories = ['Surprised', 'Fearful', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral'];
const imageCounts = [1290, 281, 717, 4772, 1982, 705, 2524];
const adjacencyCounts = [1165, 241, 256, 1915, 664, 524, 2127];

// Set intervals for refreshing data
window.onload = function () {
    updateCurrentModelDisplay();
    setInterval(refreshGradCam, 1000);
    setInterval(updateProbabilities, 2000);

    const ctx = document.getElementById('dataChart').getContext('2d');
    const dataChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: categories.map(cat => `${cat}`),
            datasets: [
                {
                    label: 'Image Count',
                    data: imageCounts,
                    backgroundColor: 'rgba(75, 192, 192, 0.6)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                },
                {
                    label: 'Adjacency Count',
                    data: adjacencyCounts,
                    backgroundColor: 'rgba(153, 102, 255, 0.6)',
                    borderColor: 'rgba(153, 102, 255, 1)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Category'
                    }
                },
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Count'
                    }
                }
            }
        }
    });

};
