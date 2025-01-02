function drawCategoryBarChart() {
    const categories = ['Surprised', 'Fearful', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral'];
    const imageCounts = [1290, 281, 717, 4772, 1982, 705, 2524];
    const adjacencyCounts = [1165, 241, 256, 1915, 664, 524, 2127];

    const ctx = document.getElementById('dataChart').getContext('2d');
    new Chart(ctx, {
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
}

function fetchAndPlotTrainAccuracy() {
    fetch('/get_train_accuracy')
        .then(response => response.json())
        .then(data => {
            const ctx = document.getElementById('trainAccuracyChart').getContext('2d');
            const datasets = data.map(item => ({
                label: item.run,
                data: item.steps.map((step, index) => ({ x: step, y: item.values[index] })),
                borderColor: `hsl(${Math.random() * 360}, 70%, 50%)`,
                borderWidth: 2,
                fill: false,
            }));

            new Chart(ctx, {
                type: 'line',
                data: {
                    datasets: datasets,
                },
                options: {
                    responsive: true,
                    scales: {
                        x: {
                            type: 'linear',
                            title: {
                                display: true,
                                text: 'Steps',
                            },
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Accuracy',
                            },
                            beginAtZero: true,
                            max: 1.0,
                        },
                    },
                    plugins: {
                        legend: {
                            position: 'top',
                        },
                    },
                },
            });
        })
        .catch(error => console.error('Error fetching train accuracy data:', error));
}

function fetchAndPlotTrainLoss() {
    fetch('/get_train_loss')
        .then(response => response.json())
        .then(data => {
            const ctx = document.getElementById('trainLossChart').getContext('2d');
            const datasets = data.map(item => ({
                label: item.run,
                data: item.steps.map((step, index) => ({ x: step, y: item.values[index] })),
                borderColor: `hsl(${Math.random() * 360}, 70%, 50%)`,
                borderWidth: 2,
                fill: false,
            }));

            new Chart(ctx, {
                type: 'line',
                data: {
                    datasets: datasets,
                },
                options: {
                    responsive: true,
                    scales: {
                        x: {
                            type: 'linear',
                            title: {
                                display: true,
                                text: 'Steps',
                            },
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Loss',
                            },
                            beginAtZero: true,
                        },
                    },
                    plugins: {
                        legend: {
                            position: 'top',
                        },
                    },
                },
            });
        })
        .catch(error => console.error('Error fetching train loss data:', error));
}

function fetchAndPlotValCorrect() {
    fetch('/get_val_correct')
        .then(response => response.json())
        .then(data => {
            const ctx = document.getElementById('valCorrectChart').getContext('2d');
            const datasets = data.map(item => ({
                label: item.run,
                data: item.steps.map((step, index) => ({ x: step, y: item.values[index] })),
                borderColor: `hsl(${Math.random() * 360}, 70%, 50%)`,
                borderWidth: 2,
                fill: false,
            }));

            new Chart(ctx, {
                type: 'line',
                data: {
                    datasets: datasets,
                },
                options: {
                    responsive: true,
                    scales: {
                        x: {
                            type: 'linear',
                            title: {
                                display: true,
                                text: 'Steps',
                            },
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Correct (%)',
                            },
                            beginAtZero: true,
                            max: 1,
                        },
                    },
                    plugins: {
                        legend: {
                            position: 'top',
                        },
                    },
                },
            });
        })
        .catch(error => console.error('Error fetching validation correct data:', error));
}

function fetchAndPlotValLoss() {
    fetch('/get_val_loss')
        .then(response => response.json())
        .then(data => {
            const ctx = document.getElementById('valLossChart').getContext('2d');
            const datasets = data.map(item => ({
                label: item.run,
                data: item.steps.map((step, index) => ({ x: step, y: item.values[index] })),
                borderColor: `hsl(${Math.random() * 360}, 70%, 50%)`,
                borderWidth: 2,
                fill: false,
            }));

            new Chart(ctx, {
                type: 'line',
                data: {
                    datasets: datasets,
                },
                options: {
                    responsive: true,
                    scales: {
                        x: {
                            type: 'linear',
                            title: {
                                display: true,
                                text: 'Steps',
                            },
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Loss',
                            },
                            beginAtZero: true,
                        },
                    },
                    plugins: {
                        legend: {
                            position: 'top',
                        },
                    },
                },
            });
        })
        .catch(error => console.error('Error fetching validation loss data:', error));
}

function fetchAndPlotParallelCoordinates() {
    fetch('/get_model_performance')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            const categories = ['train_accuracy', 'val_correct', 'lr', 'num_epochs'];

            const normalizedData = data.map(item => {
                const normalizedItem = {};
                categories.forEach(category => {
                    const values = data.map(d => d[category]);
                    const minValue = Math.min(...values);
                    const maxValue = Math.max(...values);
                    normalizedItem[category] = (item[category] - minValue) / (maxValue - minValue);
                });
                normalizedItem['Run'] = item['Run'];
                return normalizedItem;
            });

            const dimensions = categories.map(category => ({
                label: category,
                values: normalizedData.map(d => d[category]),
            }));

            const trace = {
                type: 'parcoords',
                line: {
                    color: normalizedData.map(d => d['train_accuracy']), 
                    colorscale: 'Viridis',
                },
                dimensions: dimensions,
            };

            const layout = {
                title: 'Parallel Coordinates Plot',
                width: 1000,
                height: 500,
            };

            Plotly.newPlot('parallelCoordinatesContainer', [trace], layout);
        })
        .catch(error => {
            console.error('Error fetching model performance data:', error);
            alert('Failed to load data: ' + error.message);
        });
}

window.onload = function () {
    drawCategoryBarChart();
    fetchAndPlotTrainAccuracy();
    fetchAndPlotTrainLoss();
    fetchAndPlotValCorrect();
    fetchAndPlotValLoss();
    fetchAndPlotParallelCoordinates();
};