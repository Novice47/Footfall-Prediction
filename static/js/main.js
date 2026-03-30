document.addEventListener('DOMContentLoaded', () => {
    // --- CUSTOM CURSOR ---
    const cursor = document.getElementById('cursor');
    const dot = document.getElementById('cursor-dot');
    document.addEventListener('mousemove', (e) => {
        gsap.to(cursor, { x: e.clientX - 10, y: e.clientY - 10, duration: 0.1 });
        gsap.to(dot, { x: e.clientX - 2, y: e.clientY - 2, duration: 0.05 });
    });

    // --- GSAP ANIMATIONS ---
    gsap.from(".hero h1", { duration: 1.5, y: 100, opacity: 0, ease: "power4.out" });
    new Typed('#tagline', {
        strings: ['NEURAL NETWORK FOOTFALL PREDICTION^1000', 'REAL-TIME SIGNAL ANALYSIS^1000', '10 WORLD WONDERS TRACKED^1000'],
        typeSpeed: 50,
        backSpeed: 30,
        loop: true,
        showCursor: false
    });

    // --- LEAFLET MAP ---
    const map = L.map('map', { center: [20, 0], zoom: 2, zoomControl: false });
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: 'Map tiles by CartoDB'
    }).addTo(map);

    function initMap() {
        fetch('/map-data')
            .then(res => res.json())
            .then(data => {
                L.geoJSON(data, {
                    pointToLayer: (feature, latlng) => {
                        const marker = L.circleMarker(latlng, {
                            radius: 8,
                            fillColor: getDensityColor(feature.properties.density),
                            color: "#000",
                            weight: 1,
                            opacity: 1,
                            fillOpacity: 0.8
                        });
                        marker.on('click', () => selectSite(feature.properties));
                        return marker;
                    }
                }).addTo(map);
            });
    }

    function getDensityColor(density) {
        if (density === 'high') return '#ff6b35';
        if (density === 'moderate') return '#7b2fff';
        return '#00d4ff';
    }

    // --- CHARTS ---
    let charts = {};
    function initCharts() {
        fetch('/analytics')
            .then(res => res.json())
            .then(data => {
                // Hourly Trend
                charts.hourly = new Chart(document.getElementById('hourlyChart'), {
                    type: 'line',
                    data: {
                        labels: Object.keys(data.hourly_trend),
                        datasets: [{
                            label: 'Avg Footfall',
                            data: Object.values(data.hourly_trend),
                            borderColor: '#00d4ff',
                            tension: 0.4
                        }]
                    },
                    options: chartOptions
                });

                // Site Comparison
                charts.site = new Chart(document.getElementById('siteChart'), {
                    type: 'bar',
                    data: {
                        labels: Object.keys(data.site_comparison),
                        datasets: [{
                            label: 'Mean Footfall',
                            data: Object.values(data.site_comparison),
                            backgroundColor: '#7b2fff'
                        }]
                    },
                    options: chartOptions
                });
            });
    }

    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: { grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { color: '#fff' } },
            x: { grid: { display: false }, ticks: { color: '#fff' } }
        },
        plugins: { legend: { display: false } }
    };

    // --- PREDICTION LOGIC ---
    const predictBtn = document.getElementById('predictBtn');
    predictBtn.addEventListener('click', () => {
        const payload = {
            place_id: document.getElementById('siteSelect').value,
            hour: document.getElementById('hourRange').value,
            temperature: document.getElementById('tempRange').value,
            humidity: document.getElementById('humRange').value,
            mobile_signals: document.getElementById('signalsRange').value,
            wifi_connections: document.getElementById('wifiRange').value,
            model_type: document.getElementById('modelSelect').value
        };

        fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
            .then(res => res.json())
            .then(res => {
                animateNumber('predictionVal', res.prediction);
                document.getElementById('confidenceVal').innerText = `±${res.confidence}`;
                updateTrendChart(res.hourly_trend);
                showToast('Prediction generated successfully!');
            });
    });

    function animateNumber(id, end) {
        const obj = document.getElementById(id);
        let start = 0;
        const duration = 1000;
        const startTime = performance.now();
        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            obj.innerText = Math.floor(progress * end);
            if (progress < 1) requestAnimationFrame(update);
        }
        requestAnimationFrame(update);
    }

    function updateTrendChart(trend) {
        if (charts.predict) charts.predict.destroy();
        charts.predict = new Chart(document.getElementById('predictionChart'), {
            type: 'line',
            data: {
                labels: trend.map(t => t.hour),
                datasets: [{
                    label: 'Hourly Trend',
                    data: trend.map(t => t.footfall),
                    borderColor: '#ff6b35',
                    tension: 0.4
                }]
            },
            options: chartOptions
        });
    }

    // --- DATASET EXPLORER ---
    let currentPage = 1;
    function loadDataset(page) {
        fetch(`/dataset?page=${page}`)
            .then(res => res.json())
            .then(res => {
                const body = document.getElementById('datasetBody');
                body.innerHTML = '';
                res.data.forEach(row => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td>${row.place_name}</td>
                        <td>${row.date}</td>
                        <td>${row.hour}:00</td>
                        <td>${row.footfall}</td>
                        <td>${row.temperature ? row.temperature.toFixed(1) : 'N/A'}°C</td>
                        <td>${row.mobile_signals}</td>
                    `;
                    body.appendChild(tr);
                });
                document.getElementById('pageInfo').innerText = `PAGE ${res.page}`;
                currentPage = res.page;
            });
    }

    window.loadNextPage = () => loadDataset(currentPage + 1);
    window.loadPreviousPage = () => loadDataset(Math.max(1, currentPage - 1));

    // --- MODEL METRICS ---
    function initModelMetrics() {
        fetch('/model-metrics')
            .then(res => res.json())
            .then(res => {
                const grid = document.getElementById('modelMetricsGrid');
                grid.innerHTML = '';
                for (const [name, m] of Object.entries(res.metrics)) {
                    const card = document.createElement('div');
                    card.className = 'glass-card';
                    card.innerHTML = `
                        <h3>${name.replace('_', ' ').toUpperCase()}</h3>
                        <p>R² SCORE: <span style="color: var(--primary-color)">${m.r2.toFixed(4)}</span></p>
                        <p>MAE: <span style="color: var(--secondary-color)">${m.mae.toFixed(0)}</span></p>
                        <div class="chart-container"><canvas id="metricsChart_${name}"></canvas></div>
                    `;
                    grid.appendChild(card);

                    new Chart(document.getElementById(`metricsChart_${name}`), {
                        type: 'scatter',
                        data: {
                            datasets: [{
                                label: 'Pred vs Actual',
                                data: m.predictions.map((p, i) => ({ x: m.actuals[i], y: p })),
                                backgroundColor: name === 'random_forest' ? '#00d4ff' : '#7b2fff'
                            }]
                        },
                        options: chartOptions
                    });
                }
            });
    }

    // --- SVG ANIMATION ---
    function initVisualizer() {
        const waves = document.getElementById('waves');
        setInterval(() => {
            const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
            circle.setAttribute("cx", "100");
            circle.setAttribute("cy", "150");
            circle.setAttribute("r", "5");
            circle.setAttribute("fill", "none");
            circle.setAttribute("stroke", "var(--primary-color)");
            waves.appendChild(circle);

            gsap.to(circle, {
                attr: { r: 100 },
                opacity: 0,
                duration: 2,
                onComplete: () => waves.removeChild(circle)
            });

            // Pulse to model
            const pulse = document.createElementNS("http://www.w3.org/2000/svg", "circle");
            pulse.setAttribute("cx", "100");
            pulse.setAttribute("cy", "150");
            pulse.setAttribute("r", "5");
            pulse.setAttribute("fill", "var(--primary-color)");
            waves.appendChild(pulse);

            gsap.to(pulse, {
                x: 250,
                duration: 1,
                opacity: 0,
                onComplete: () => waves.removeChild(pulse)
            });
        }, 1000);
    }

    // Initialize
    initMap();
    initCharts();
    loadDataset(1);
    initModelMetrics();
    initVisualizer();
});
