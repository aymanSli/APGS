#!/usr/bin/env python3
# app.py - Application Web pour la Simulation de Produit Structuré
from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import sys
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Import de notre simulation
from main import StructuredProductSimulation

app = Flask(__name__)

# Variable globale pour stocker l'instance de la simulation
simulation = None
portfolio_history = []


@app.route('/')
def index():
    """Page d'accueil de l'application."""
    if simulation is None:
        return redirect(url_for('initialize'))

    # Obtenir les données du portefeuille
    portfolio_data = get_portfolio_data()

    # Obtenir les données du produit
    product_data = get_product_data()

    # Obtenir l'historique des dividendes
    dividends_data = get_dividends_data()

    # Obtenir l'historique des flux
    flows_data = get_flows_data()

    return render_template(
        'index.html',
        date=simulation.current_date.strftime('%Y-%m-%d'),
        portfolio=portfolio_data,
        product=product_data,
        dividends=dividends_data,
        flows=flows_data
    )


@app.route('/initialize', methods=['GET', 'POST'])
def initialize():
    """Page d'initialisation de la simulation."""
    if request.method == 'POST':
        # Initialiser la simulation avec le fichier par défaut
        global simulation
        data_file = "DonneesGPS2025.xlsx"

        # Vérifier si le fichier existe
        if not os.path.exists(data_file):
            return render_template('initialize.html', error=f"Fichier '{data_file}' introuvable.")

        # Initialiser la simulation
        try:
            simulation = StructuredProductSimulation(data_file)

            # Initialiser l'historique du portefeuille
            global portfolio_history
            portfolio_history = []

            # Ajouter la valeur initiale
            spot_prices = simulation.past_data.get_spot_prices()
            portfolio_state = simulation.portfolio.get_portfolio_state(spot_prices)
            portfolio_history.append({
                'date': simulation.current_date.strftime('%Y-%m-%d'),
                'value': portfolio_state['total_value']
            })

            return redirect(url_for('index'))

        except Exception as e:
            return render_template('initialize.html', error=f"Erreur d'initialisation: {str(e)}")

    return render_template('initialize.html')


@app.route('/advance', methods=['POST'])
def advance():
    """Avancer la simulation d'un nombre de jours spécifié."""
    if simulation is None:
        return jsonify({'success': False, 'error': 'Simulation non initialisée'}), 400

    days = request.form.get('days', type=int)
    if days <= 0:
        return jsonify({'success': False, 'error': 'Le nombre de jours doit être positif'}), 400

    try:
        # Avancer la simulation
        simulation.advance_days(days)

        # Mettre à jour l'historique du portefeuille
        spot_prices = simulation.past_data.get_spot_prices()
        portfolio_state = simulation.portfolio.get_portfolio_state(spot_prices)
        portfolio_history.append({
            'date': simulation.current_date.strftime('%Y-%m-%d'),
            'value': portfolio_state['total_value']
        })

        return jsonify({
            'success': True,
            'date': simulation.current_date.strftime('%Y-%m-%d'),
            'portfolio': get_portfolio_data(),
            'product': get_product_data(),
            'dividends': get_dividends_data(),
            'flows': get_flows_data()
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/jump_key_date', methods=['POST'])
def jump_key_date():
    """Sauter à la prochaine date clé."""
    if simulation is None:
        return jsonify({'success': False, 'error': 'Simulation non initialisée'}), 400

    try:
        # Sauter à la prochaine date clé
        simulation.jump_to_next_key_date()

        # Mettre à jour l'historique du portefeuille
        spot_prices = simulation.past_data.get_spot_prices()
        portfolio_state = simulation.portfolio.get_portfolio_state(spot_prices)
        portfolio_history.append({
            'date': simulation.current_date.strftime('%Y-%m-%d'),
            'value': portfolio_state['total_value']
        })

        return jsonify({
            'success': True,
            'date': simulation.current_date.strftime('%Y-%m-%d'),
            'portfolio': get_portfolio_data(),
            'product': get_product_data(),
            'dividends': get_dividends_data(),
            'flows': get_flows_data()
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/get_rebalancing_info', methods=['GET'])
def get_rebalancing_info():
    """Obtenir les informations nécessaires pour le rebalancement."""
    if simulation is None:
        return jsonify({'success': False, 'error': 'Simulation non initialisée'}), 400

    try:
        # Obtenir les données du portefeuille
        spot_prices = simulation.past_data.get_spot_prices()
        portfolio_state = simulation.portfolio.get_portfolio_state(spot_prices)

        # Calculer les deltas théoriques (ceux qui seraient appliqués lors d'un rebalancement)
        theoretical_deltas = simulation.monte_carlo.calculate_deltas(
            simulation.past_matrix,
            simulation.current_date
        )

        # Préparer les données de position avec les sensibilités
        positions = []
        for i in range(len(simulation.portfolio.column_names)):
            current_delta = simulation.portfolio.deltas[i]
            theo_delta = theoretical_deltas[i]
            sensitivity = theo_delta - current_delta  # Différence entre delta théorique et actuel

            positions.append({
                'name': simulation.portfolio.column_names[i],
                'delta': current_delta,
                'theoretical_delta': theo_delta,
                'sensitivity': sensitivity,
                'price': spot_prices[i],
                'value': portfolio_state['position_values'][i]
            })

        # Calculer des mesures de risque (exemples)
        risk_measures = {
            'total_delta': sum(simulation.portfolio.deltas),
            'gamma': 0.01,  # Exemple, à remplacer par vrai calcul
            'vega': 0.02  # Exemple, à remplacer par vrai calcul
        }

        return jsonify({
            'success': True,
            'positions': positions,
            'risk_measures': risk_measures,
            'cash': portfolio_state['cash'],
            'total_value': portfolio_state['total_value']
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/rebalance', methods=['POST'])
def rebalance():
    """Rebalancer le portefeuille."""
    if simulation is None:
        return jsonify({'success': False, 'error': 'Simulation non initialisée'}), 400

    try:
        # Sauvegarder les anciens deltas
        old_deltas = simulation.portfolio.deltas.copy()

        # Rebalancer le portefeuille
        rebalance_result = simulation.rebalance_portfolio()

        # Préparer les détails des transactions
        trades = []
        for i, (old_delta, new_delta) in enumerate(zip(old_deltas, simulation.portfolio.deltas)):
            asset_name = simulation.portfolio.column_names[i]
            delta_change = new_delta - old_delta

            trades.append({
                'asset': asset_name,
                'old_delta': old_delta,
                'new_delta': new_delta,
                'delta_change': delta_change,
                'value': rebalance_result.get('position_values', [0] * len(old_deltas))[
                    i] if 'position_values' in rebalance_result else 0
            })

        # Préparer les détails du rebalancement
        rebalance_details = {
            'trades': trades,
            'cash_before': rebalance_result.get('initial_cash', 0),
            'cash_after': rebalance_result.get('cash', 0),
            'total_value_before': rebalance_result.get('initial_value', 0),
            'total_value_after': rebalance_result.get('final_value', 0),
            'pnl': rebalance_result.get('pnl', 0)
        }

        return jsonify({
            'success': True,
            'portfolio': get_portfolio_data(),
            'product': get_product_data(),
            'rebalance_details': rebalance_details
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
@app.route('/product_details', methods=['GET'])
def product_details():
    """Obtenir les détails du produit."""
    if simulation is None:
        return jsonify({'success': False, 'error': 'Simulation non initialisée'}), 400

    try:
        # Simuler le cycle de vie du produit
        lifecycle = simulation.product.simulate_product_lifecycle(
            simulation.past_matrix,
            simulation.current_date
        )

        # Convertir les dates en chaînes
        formatted_lifecycle = []
        for event in lifecycle:
            formatted_event = event.copy()
            if 'date' in formatted_event:
                formatted_event['date'] = formatted_event['date'].strftime('%Y-%m-%d')
            formatted_lifecycle.append(formatted_event)

        return jsonify({
            'success': True,
            'lifecycle': formatted_lifecycle
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/set_date', methods=['POST'])
def set_date():
    """Définir une date spécifique pour la simulation."""
    if simulation is None:
        return jsonify({'success': False, 'error': 'Simulation non initialisée'}), 400

    date_str = request.form.get('date')
    if not date_str:
        return jsonify({'success': False, 'error': 'Date non spécifiée'}), 400

    try:
        # Convertir la chaîne de date en objet datetime
        target_date = datetime.strptime(date_str, '%Y-%m-%d')

        # Vérifier si la date est valide
        if target_date < simulation.date_handler.key_dates['T0']:
            return jsonify({'success': False, 'error': 'La date ne peut pas être antérieure à T0'}), 400

        if target_date > simulation.date_handler.key_dates['Tc']:
            return jsonify({'success': False, 'error': 'La date ne peut pas être postérieure à Tc'}), 400

        # Trouver l'index de date correspondant
        current_index = simulation.date_handler.get_date_index(simulation.current_date)
        target_index = simulation.date_handler.get_date_index(target_date)

        # Calculer le nombre de jours à avancer
        days_to_advance = target_index - current_index

        if days_to_advance == 0:
            return jsonify({'success': True, 'message': 'Déjà à cette date'}), 200

        # Avancer la simulation
        if days_to_advance > 0:
            simulation.advance_days(days_to_advance)
        else:
            # Pour reculer dans le temps, il faudrait réinitialiser et avancer jusqu'à la date cible
            # Ce n'est pas implémenté dans la simulation actuelle
            return jsonify({'success': False, 'error': 'Impossible de reculer dans le temps'}), 400

        # Mettre à jour l'historique du portefeuille
        spot_prices = simulation.past_data.get_spot_prices()
        portfolio_state = simulation.portfolio.get_portfolio_state(spot_prices)
        portfolio_history.append({
            'date': simulation.current_date.strftime('%Y-%m-%d'),
            'value': portfolio_state['total_value']
        })

        return jsonify({
            'success': True,
            'date': simulation.current_date.strftime('%Y-%m-%d'),
            'portfolio': get_portfolio_data(),
            'product': get_product_data(),
            'dividends': get_dividends_data(),
            'flows': get_flows_data()
        })

    except ValueError:
        return jsonify({'success': False, 'error': 'Format de date invalide. Utilisez YYYY-MM-DD'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def get_portfolio_data():
    """Obtenir les données actuelles du portefeuille."""
    spot_prices = simulation.past_data.get_spot_prices()
    portfolio_state = simulation.portfolio.get_portfolio_state(spot_prices)

    # Positions détaillées
    positions = []
    for i in range(len(simulation.portfolio.column_names)):  # Utiliser la longueur réelle de column_names
        price = spot_prices[i]
        position_value = portfolio_state['position_values'][i]

        # Déterminer la devise
        currency = "€"  # Par défaut, l'euro
        foreign_currency = None

        # Indice du produit
        index_name = simulation.portfolio.column_names[i]

        # Pour les devises, on a un prix en devise étrangère et en euro
        if index_name in ["RUSD", "RGBP", "RJPY", "RHKD"]:
            if index_name == "RUSD":
                foreign_currency = "$"
            elif index_name == "RGBP":
                foreign_currency = "£"
            elif index_name == "RJPY":
                foreign_currency = "¥"
            elif index_name == "RHKD":
                foreign_currency = "HK$"

        positions.append({
            'name': index_name,
            'delta': simulation.portfolio.deltas[i],
            'price': price,
            'value': position_value,
            'currency': currency,
            'foreign_currency': foreign_currency
        })

    # Valeurs globales du portefeuille
    portfolio_data = {
        'cash': portfolio_state['cash'],
        'total_position_value': portfolio_state['total_position_value'],
        'total_value': portfolio_state['total_value'],
        'interest_rate': simulation.risk_free_rate * 100,  # En pourcentage
        'positions': positions,
        'pnl': calculate_pnl()
    }

    return portfolio_data


def get_product_data():
    """Obtenir les données actuelles du produit."""
    product_data = {
        'excluded_indices': simulation.excluded_indices,
        'guarantee_triggered': simulation.guarantee_triggered,
        'next_key_date': get_next_key_date_info()
    }

    return product_data


def get_dividends_data():
    """Obtenir les données des dividendes."""
    dividends = []
    for div in simulation.dividends_paid:
        dividends.append({
            'date': div['date'].strftime('%Y-%m-%d'),
            'key': div['key'],
            'amount': div['amount'],
            'index': div['index'],
            'return': div['return'] * 100  # En pourcentage
        })

    return dividends


def get_flows_data():
    """Obtenir les données des flux financiers."""
    return portfolio_history


def get_next_key_date_info():
    """Obtenir les informations sur la prochaine date clé."""
    next_key_date = simulation.date_handler.get_next_key_date(simulation.current_date)

    if not next_key_date:
        return {
            'exists': False,
            'message': "Aucune date clé future"
        }

    # Trouver le nom de la clé
    key_name = None
    for key, date in simulation.date_handler.key_dates.items():
        if date == next_key_date:
            key_name = key
            break

    days_to_next = simulation.date_handler._count_trading_days(
        simulation.current_date,
        next_key_date
    )

    return {
        'exists': True,
        'date': next_key_date.strftime('%Y-%m-%d'),
        'key': key_name,
        'days_remaining': days_to_next
    }


def calculate_pnl():
    """Calculer le P&L depuis le début de la simulation."""
    if len(portfolio_history) < 2:
        return 0.0

    initial_value = portfolio_history[0]['value']
    current_value = portfolio_history[-1]['value']

    return ((current_value / initial_value) - 1) * 100  # P&L en pourcentage


if __name__ == '__main__':
    app.run(debug=True, port=8080)