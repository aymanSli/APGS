// Script pour l'interface de gestion de produit structuré
document.addEventListener('DOMContentLoaded', function() {
    // Initialiser le graphique de flux
    initFlowsChart();

    // Gestionnaire pour le bouton d'avancement
    document.getElementById('advance-btn').addEventListener('click', function() {
        const days = document.getElementById('days-input').value;
        if (days <= 0) {
            alert("Le nombre de jours doit être positif.");
            return;
        }

        advanceDays(days);
    });

    // Gestionnaire pour le bouton de saut à la date clé suivante
    document.getElementById('key-date-btn').addEventListener('click', function() {
        jumpToKeyDate();
    });

    // Gestionnaire pour le bouton de rebalancement
    document.getElementById('rebalance-btn').addEventListener('click', function() {
        rebalancePortfolio();
    });

    // Gestionnaire pour le bouton d'informations sur le rebalancement
    document.getElementById('rebalancing-info-btn').addEventListener('click', function() {
        showRebalancingInfo();
    });

    // Gestionnaire pour le bouton de détails du produit
    document.getElementById('product-details-btn').addEventListener('click', function() {
        showProductDetails();
    });

    // Gestionnaire pour le sélecteur de date
    document.getElementById('date-picker').addEventListener('change', function() {
        const selectedDate = this.value;
        if (selectedDate) {
            setSimulationDate(selectedDate);
        }
    });

    // Initialiser le sélecteur de date avec la date actuelle
    const currentDate = document.getElementById('current-date').innerText;
    document.getElementById('date-picker').value = currentDate;
});

// Graphique de flux
let flowsChart = null;

function initFlowsChart() {
    const ctx = document.getElementById('flows-chart').getContext('2d');

    // Récupérer les données de flux depuis la page
    const flowsData = JSON.parse('{{ flows|tojson }}');

    const labels = flowsData.map(item => item.date);
    const values = flowsData.map(item => item.value);

    flowsChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Valeur du portefeuille (€)',
                data: values,
                borderColor: 'rgba(13, 110, 253, 1)',
                backgroundColor: 'rgba(13, 110, 253, 0.1)',
                borderWidth: 2,
                tension: 0.1,
                fill: true
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            return `Valeur: ${context.parsed.y.toFixed(2)} €`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    title: {
                        display: true,
                        text: 'Valeur (€)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Date'
                    }
                }
            }
        }
    });
}

function updateFlowsChart(flowsData) {
    const labels = flowsData.map(item => item.date);
    const values = flowsData.map(item => item.value);

    flowsChart.data.labels = labels;
    flowsChart.data.datasets[0].data = values;
    flowsChart.update();
}

// Fonctions pour les actions de l'interface
function advanceDays(days) {
    // Désactiver les boutons pendant le traitement
    setLoading(true);

    // Envoyer la requête AJAX
    fetch('/advance', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `days=${days}`
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            updateInterface(data);
        } else {
            alert(`Erreur: ${data.error}`);
        }
    })
    .catch(error => {
        console.error('Erreur:', error);
        alert('Une erreur est survenue lors de l\'avancement de la simulation.');
    })
    .finally(() => {
        setLoading(false);
    });
}

function jumpToKeyDate() {
    // Désactiver les boutons pendant le traitement
    setLoading(true);

    // Envoyer la requête AJAX
    fetch('/jump_key_date', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            updateInterface(data);
        } else {
            alert(`Erreur: ${data.error}`);
        }
    })
    .catch(error => {
        console.error('Erreur:', error);
        alert('Une erreur est survenue lors du saut à la date clé suivante.');
    })
    .finally(() => {
        setLoading(false);
    });
}

function rebalancePortfolio() {
    // Désactiver les boutons pendant le traitement
    setLoading(true);

    // Envoyer la requête AJAX
    fetch('/rebalance', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Mettre à jour uniquement les parties du portefeuille
            updatePortfolioInfo(data.portfolio);
            updateProductInfo(data.product);
            alert('Portefeuille rebalancé avec succès!');
        } else {
            alert(`Erreur: ${data.error}`);
        }
    })
    .catch(error => {
        console.error('Erreur:', error);
        alert('Une erreur est survenue lors du rebalancement du portefeuille.');
    })
    .finally(() => {
        setLoading(false);
    });
}

function showRebalancingInfo() {
    alert('Information de rebalancement: Cette fonctionnalité affichera des informations détaillées sur le rebalancement du portefeuille.');
}

function showProductDetails() {
    // Envoyer la requête AJAX
    fetch('/product_details')
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Créer une représentation visuelle du cycle de vie du produit
            const lifecycle = data.lifecycle;
            let htmlContent = '<div class="table-responsive"><table class="table table-striped">';
            htmlContent += '<thead><tr><th>Date</th><th>Événement</th><th>Détails</th></tr></thead><tbody>';

            lifecycle.forEach(event => {
                htmlContent += `<tr>
                    <td>${event.date || '-'}</td>
                    <td>${event.type || event.name || '-'}</td>
                    <td>${event.description || '-'}</td>
                </tr>`;
            });

            htmlContent += '</tbody></table></div>';

            // Créer une boîte de dialogue modale
            const modal = new bootstrap.Modal(document.getElementById('product-details-modal'));
            document.querySelector('#product-details-modal .modal-body').innerHTML = htmlContent;
            modal.show();
        } else {
            alert(`Erreur: ${data.error}`);
        }
    })
    .catch(error => {
        console.error('Erreur:', error);
        alert('Une erreur est survenue lors de la récupération des détails du produit.');
    });
}

function setSimulationDate(dateStr) {
    // Désactiver les boutons pendant le traitement
    setLoading(true);

    // Envoyer la requête AJAX
    fetch('/set_date', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `date=${dateStr}`
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            updateInterface(data);
        } else {
            alert(`Erreur: ${data.error}`);
        }
    })
    .catch(error => {
        console.error('Erreur:', error);
        alert('Une erreur est survenue lors du changement de date.');
    })
    .finally(() => {
        setLoading(false);
    });
}

// Fonction pour désactiver/activer les boutons pendant le chargement
function setLoading(isLoading) {
    const buttons = document.querySelectorAll('button');
    buttons.forEach(button => {
        button.disabled = isLoading;
    });

    const inputs = document.querySelectorAll('input');
    inputs.forEach(input => {
        input.disabled = isLoading;
    });
}

// Fonctions pour mettre à jour l'interface
function updateInterface(data) {
    // Mettre à jour la date
    document.getElementById('current-date').innerText = data.date;
    document.getElementById('date-picker').value = data.date;

    // Mettre à jour les informations du portefeuille
    updatePortfolioInfo(data.portfolio);

    // Mettre à jour les informations du produit
    updateProductInfo(data.product);

    // Mettre à jour le tableau des dividendes
    updateDividendsTable(data.dividends);

    // Mettre à jour le graphique des flux
    updateFlowsChart(data.flows);
}

function updatePortfolioInfo(portfolio) {
    // Mettre à jour le tableau des positions
    const positionsTable = document.getElementById('positions-table');
    positionsTable.innerHTML = '';

    portfolio.positions.forEach(position => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${position.name}</td>
            <td class="text-end">${position.delta.toFixed(6)}</td>
            <td class="text-end">${position.price.toFixed(6)} €</td>
            <td class="text-end">
                ${position.foreign_currency ? `${position.price.toFixed(6)} ${position.foreign_currency}` : '-'}
            </td>
            <td class="text-end">${position.value.toFixed(2)} €</td>
        `;
        positionsTable.appendChild(row);
    });

    // Mettre à jour les valeurs du portefeuille
    document.querySelector('tfoot th:nth-child(2)').innerText = `${portfolio.cash.toFixed(2)} €`;
    document.querySelector('tfoot tr:nth-child(2) th:nth-child(2)').innerText = `${portfolio.interest_rate.toFixed(2)} %`;
    document.querySelector('tfoot tr:nth-child(3) th:nth-child(2)').innerText = `${portfolio.total_value.toFixed(2)} €`;

    // Mettre à jour le P&L
    const pnlElement = document.querySelector('.card-header .badge');
    pnlElement.className = `badge ${portfolio.pnl >= 0 ? 'bg-success' : 'bg-danger'}`;
    pnlElement.innerText = `${portfolio.pnl.toFixed(2)}%`;

    const pnlValueElement = document.querySelector('.card-body h2');
    pnlValueElement.innerText = `${portfolio.pnl.toFixed(2)}%`;
    pnlValueElement.parentElement.className = `text-center ${portfolio.pnl >= 0 ? 'bg-profit' : 'bg-loss'} p-4 rounded`;

    // Mettre à jour la valeur totale du portefeuille
    document.querySelector('.card-body h2:nth-of-type(2)').innerText = `${portfolio.total_value.toFixed(2)} €`;
}

function updateProductInfo(product) {
    // Mettre à jour les indices exclus
    const excludedIndicesElement = document.getElementById('excluded-indices');
    excludedIndicesElement.innerText = product.excluded_indices.length > 0
        ? product.excluded_indices.join(', ')
        : 'Aucun';

    // Mettre à jour le statut de la garantie
    const guaranteeStatusElement = document.getElementById('guarantee-status');
    guaranteeStatusElement.innerHTML = product.guarantee_triggered
        ? '<span class="badge bg-success">Oui</span>'
        : '<span class="badge bg-warning">Non</span>';

    // Mettre à jour les informations sur la prochaine date clé
    const nextKeyDateInfoElement = document.getElementById('next-key-date-info');
    if (product.next_key_date.exists) {
        nextKeyDateInfoElement.innerHTML = `
            <small class="text-muted">
                Prochaine date clé: <strong>${product.next_key_date.key}</strong>
                (${product.next_key_date.date}) -
                <span class="badge bg-info">${product.next_key_date.days_remaining} jours restants</span>
            </small>
        `;
    } else {
        nextKeyDateInfoElement.innerHTML = `
            <small class="text-muted">${product.next_key_date.message}</small>
        `;
    }
}

function updateDividendsTable(dividends) {
    const dividendsTable = document.getElementById('dividends-table').querySelector('tbody');
    dividendsTable.innerHTML = '';

    if (dividends && dividends.length > 0) {
        dividends.forEach(dividend => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${dividend.date}</td>
                <td>${dividend.key}</td>
                <td>${dividend.amount.toFixed(2)} €</td>
                <td>${dividend.index}</td>
                <td>${dividend.return.toFixed(2)}%</td>
            `;
            dividendsTable.appendChild(row);
        });
    } else {
        const row = document.createElement('tr');
        row.innerHTML = '<td colspan="5" class="text-center">Aucun dividende versé</td>';
        dividendsTable.appendChild(row);
    }
}