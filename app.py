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
        print(index_name)
        # Pour les devises, on a un prix en devise étrangère et en euro
        if index_name in ["ASX200", "NASDAQ100", "SMI", "FTSE100"]:
            if index_name == "NASDAQ100":
                foreign_currency = "$"
            elif index_name == "FTSE100":
                foreign_currency = "£"
            elif index_name == "SMI":
                foreign_currency = "CHF"
            elif index_name == "ASX200":
                foreign_currency = "A$"

        positions.append({
            'name': index_name,
            'delta': simulation.portfolio.deltas[i],
            'price': price,
            'price_foreign':price ,
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
        'pnl': calculate_pnl(),
        'liquidative_value':simulation.monte_carlo.price
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


@app.route('/market_info')
def market_info():
    """Page d'informations de marché."""
    if simulation is None:
        return redirect(url_for('initialize'))

    # Obtenir les données de marché pour les indices du panier
    market_data = {}

    # Date actuelle et index correspondant
    current_date = simulation.current_date
    date_index = simulation.market_data.get_date_index(current_date)

    # Date T0 et index correspondant
    t0_date = simulation.date_handler.key_dates['T0']
    t0_index = simulation.market_data.get_date_index(t0_date)

    # Date d'hier (pour calcul de performance journalière)
    yesterday_index = max(date_index - 1, 0)

    # Date d'il y a un mois (pour calcul de performance mensuelle)
    month_ago_index = max(date_index - 21, 0)  # Approximativement 21 jours de trading par mois

    # Récupérer les infos pour chaque indice
    indices = simulation.market_data.indices

    for index_name in indices:
        # Prix actuel
        current_price = simulation.market_data.get_asset_price(index_name, date_index)

        # Prix hier
        yesterday_price = simulation.market_data.get_asset_price(index_name, yesterday_index)

        # Prix il y a un mois
        month_ago_price = simulation.market_data.get_asset_price(index_name, month_ago_index)

        # Prix à T0
        t0_price = simulation.market_data.get_asset_price(index_name, t0_index)

        # Devise de l'indice
        currency = simulation.market_data.index_currencies[index_name]

        # Calculer les performances
        daily_perf = (current_price / yesterday_price) - 1
        monthly_perf = (current_price / month_ago_price) - 1
        since_t0_perf = (current_price / t0_price) - 1

        # Stocker les informations
        market_data[index_name] = {
            'current_price': current_price,
            'daily_perf': daily_perf * 100,  # En pourcentage
            'monthly_perf': monthly_perf * 100,
            'since_t0_perf': since_t0_perf * 100,
            'currency': currency
        }

    # Calculer la performance moyenne du panier
    basket_daily_perf = sum(data['daily_perf'] for data in market_data.values()) / len(indices)
    basket_monthly_perf = sum(data['monthly_perf'] for data in market_data.values()) / len(indices)
    basket_since_t0_perf = sum(data['since_t0_perf'] for data in market_data.values()) / len(indices)

    # Trouver le meilleur et le pire indice (depuis T0)
    best_index = max(indices, key=lambda idx: market_data[idx]['since_t0_perf'])
    worst_index = min(indices, key=lambda idx: market_data[idx]['since_t0_perf'])

    # Récupérer les taux de change
    fx_rates = {}
    for currency in set(simulation.market_data.index_currencies.values()):
        if currency != 'EUR':  # EUR/EUR = 1
            fx_rates[f'EUR/{currency}'] = simulation.market_data.get_exchange_rate(currency, date_index)

    # Taux d'intérêt EUR
    eur_rate = simulation.market_data.get_interest_rate('EUR', date_index) * 100  # En pourcentage

    # Récupérer les deltas et valeurs actuelles du portefeuille
    spot_prices = simulation.past_data.get_spot_prices()
    portfolio_state = simulation.portfolio.get_portfolio_state(spot_prices)

    # Calcul de l'impact sur la valeur du produit
    product_value = portfolio_state['total_value']
    product_perf = ((product_value / 1000.0) - 1) * 100  # Assumant une valeur initiale de 1000€

    basket_data = {
        'daily_perf': basket_daily_perf,
        'monthly_perf': basket_monthly_perf,
        'since_t0_perf': basket_since_t0_perf,
        'best_index': best_index,
        'best_index_perf': market_data[best_index]['since_t0_perf'],
        'worst_index': worst_index,
        'worst_index_perf': market_data[worst_index]['since_t0_perf'],
        'product_value': product_value,
        'product_perf': product_perf
    }

    # Création d'un historique des prix pour le graphique
    # (Idéalement, ceci serait basé sur des données historiques réelles)
    price_history = {}

    # Rassembler les données pour le template
    return render_template(
        'marketdata_info.html',
        date=current_date.strftime('%Y-%m-%d'),
        market_data=market_data,
        basket_data=basket_data,
        fx_rates=fx_rates,
        eur_rate=eur_rate,
        portfolio=portfolio_state
    )


def get_market_data():
    """Obtenir les données de marché pour les indices du panier."""
    # Données de marché pour les indices
    market_data = {}

    # Date actuelle et index correspondant
    current_date = simulation.current_date
    date_index = simulation.market_data.get_date_index(current_date)

    # Date T0 et index correspondant
    t0_date = simulation.date_handler.key_dates['T0']
    t0_index = simulation.market_data.get_date_index(t0_date)

    # Date d'hier (pour calcul de performance journalière)
    yesterday_index = max(date_index - 1, 0)

    # Date d'il y a un mois (pour calcul de performance mensuelle)
    month_ago_index = max(date_index - 21, 0)  # Approximativement 21 jours de trading par mois

    # Récupérer les infos pour chaque indice
    indices = simulation.market_data.indices

    for index_name in indices:
        # Prix actuel
        current_price = simulation.market_data.get_asset_price(index_name, date_index)

        # Prix hier
        yesterday_price = simulation.market_data.get_asset_price(index_name, yesterday_index)

        # Prix il y a un mois
        month_ago_price = simulation.market_data.get_asset_price(index_name, month_ago_index)

        # Prix à T0
        t0_price = simulation.market_data.get_asset_price(index_name, t0_index)

        # Devise de l'indice
        currency = simulation.market_data.index_currencies[index_name]

        # Calculer les performances
        daily_perf = (current_price / yesterday_price) - 1
        monthly_perf = (current_price / month_ago_price) - 1
        since_t0_perf = (current_price / t0_price) - 1

        # Stocker les informations
        market_data[index_name] = {
            'current_price': current_price,
            'daily_perf': daily_perf * 100,  # En pourcentage
            'monthly_perf': monthly_perf * 100,
            'since_t0_perf': since_t0_perf * 100,
            'currency': currency
        }

    # Calculer la performance moyenne du panier
    basket_daily_perf = sum(data['daily_perf'] for data in market_data.values()) / len(indices)
    basket_monthly_perf = sum(data['monthly_perf'] for data in market_data.values()) / len(indices)
    basket_since_t0_perf = sum(data['since_t0_perf'] for data in market_data.values()) / len(indices)

    # Trouver le meilleur et le pire indice (depuis T0)
    best_index = max(indices, key=lambda idx: market_data[idx]['since_t0_perf'])
    worst_index = min(indices, key=lambda idx: market_data[idx]['since_t0_perf'])

    # Récupérer les taux de change
    fx_rates = {}
    for currency in set(simulation.market_data.index_currencies.values()):
        if currency != 'EUR':  # EUR/EUR = 1
            fx_rates[f'EUR/{currency}'] = simulation.market_data.get_exchange_rate(currency, date_index)

    # Taux d'intérêt EUR
    eur_rate = simulation.market_data.get_interest_rate('EUR', date_index) * 100  # En pourcentage

    # Calculer l'impact sur la valeur du produit (utiliser le portefeuille existant)
    portfolio_data = get_portfolio_data()  # Réutiliser cette fonction
    product_value = portfolio_data['total_value']
    product_perf = ((product_value / 1000.0) - 1) * 100  # Assumant une valeur initiale de 1000€

    basket_data = {
        'daily_perf': basket_daily_perf,
        'monthly_perf': basket_monthly_perf,
        'since_t0_perf': basket_since_t0_perf,
        'best_index': best_index,
        'best_index_perf': market_data[best_index]['since_t0_perf'],
        'worst_index': worst_index,
        'worst_index_perf': market_data[worst_index]['since_t0_perf'],
        'product_value': product_value,
        'product_perf': product_perf
    }

    return market_data, basket_data, fx_rates, eur_rate


@app.route('/pay_dividend', methods=['POST'])
def pay_dividend():
    """Payer un dividende et exclure l'indice correspondant."""
    if simulation is None:
        return jsonify({'success': False, 'error': 'Simulation non initialisée'}), 400

    try:
        # Trouver la dernière date clé que nous avons dépassée
        last_key_date, key_name = None, None
        for k, date in simulation.date_handler.key_dates.items():
            if date <= simulation.current_date and k not in ['T0', 'Tc']:  # Ignorer T0 et Tc
                if last_key_date is None or date > last_key_date:
                    last_key_date = date
                    key_name = k

        if last_key_date is None or key_name is None:
            return jsonify({'success': False, 'error': 'Aucune date clé de paiement n\'a encore été atteinte'}), 400

        # Vérifier si le dividende pour cette date clé a déjà été payé
        for div in simulation.dividends_paid:
            if div['key'] == key_name:
                return jsonify(
                    {'success': False, 'error': f'Le dividende pour la date clé {key_name} a déjà été payé'}), 400

        # Calculer les indices exclus
        exclude_indices = simulation.excluded_indices.copy()

        # Calculer le dividende
        dividend, best_index, best_return = simulation.product.calculate_dividend(
            key_name,
            simulation.past_matrix,
            exclude_indices
        )

        if dividend <= 0:
            return jsonify({'success': False, 'error': 'Aucun dividende à payer à cette date'}), 400

        # Mettre à jour les indices exclus
        if best_index and best_index not in simulation.excluded_indices:
            simulation.excluded_indices.append(best_index)

        # Vérifier si la garantie minimale est déclenchée
        guarantee_check = simulation.product.check_minimum_guarantee(key_name, simulation.past_matrix)
        if guarantee_check and not simulation.guarantee_triggered:
            simulation.guarantee_triggered = True

        # Traiter le paiement du dividende
        spot_prices = simulation.past_data.get_spot_prices()
        payment = simulation.portfolio.process_dividend_payment(
            dividend,
            simulation.current_date,
            spot_prices
        )

        # Enregistrer le paiement du dividende
        simulation.dividends_paid.append({
            'date': simulation.current_date,
            'key': key_name,
            'amount': dividend,
            'index': best_index,
            'return': best_return
        })

        return jsonify({
            'success': True,
            'dividend': dividend,
            'best_index': best_index,
            'best_return': best_return * 100,  # En pourcentage
            'excluded_indices': simulation.excluded_indices,
            'guarantee_triggered': simulation.guarantee_triggered,
            'portfolio': get_portfolio_data(),
            'dividends': get_dividends_data()
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080)