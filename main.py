# main.py
import sys
import os
import argparse
import logging

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from football_predictor.betting_predictor import BettingPredictor, print_prediction_report

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main entry point for the Football Prediction System.
    Provides a command-line interface to run predictions or initialize the system.
    """
    parser = argparse.ArgumentParser(description="Football Prediction System CLI")
    parser.add_argument(
        'action', 
        choices=['predict', 'init'], 
        help="The action to perform: 'init' to train the system, 'predict' to make a prediction."
    )
    parser.add_argument('--home', type=str, help="Home team name for prediction.")
    parser.add_argument('--away', type=str, help="Away team name for prediction.")
    
    args = parser.parse_args()

    predictor = BettingPredictor()

    if args.action == 'init':
        logging.info("Starting system initialization and training...")
        predictor.initialize()
        logging.info("System initialization complete.")
    
    elif args.action == 'predict':
        if not args.home or not args.away:
            parser.error("--home and --away arguments are required for the 'predict' action.")
        
        logging.info(f"Making a prediction for {args.home} vs {args.away}...")
        
        # Using default odds for demonstration
        # In a real scenario, these would be fetched from an API
        example_odds = {
            'over15_odds': 1.2,
            'over25_odds': 1.8,
            'over35_odds': 3.2,
            'over45_odds': 6.5
        }
        
        result = predictor.predict_match(
            home_team=args.home,
            away_team=args.away,
            **example_odds
        )
        
        print_prediction_report(result)

if __name__ == "__main__":
    main()
