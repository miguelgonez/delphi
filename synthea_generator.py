"""
Synthea-inspired Synthetic Health Data Generator for Delphi
Creates realistic but synthetic patient health trajectories compatible with Delphi's data pipeline.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import random
import uuid
from utils import get_tokenizer, get_disease_mapping

@dataclass
class PopulationConfig:
    """Configuration for synthetic population generation"""
    num_patients: int = 1000
    age_range: Tuple[int, int] = (18, 90)
    gender_distribution: Dict[str, float] = None  # {'M': 0.5, 'F': 0.5}
    geographic_region: str = "Massachusetts"  # For demographic statistics
    seed: int = 42
    start_year: int = 2000
    end_year: int = 2024
    
    def __post_init__(self):
        if self.gender_distribution is None:
            self.gender_distribution = {'M': 0.5, 'F': 0.5}

class SyntheaGenerator:
    """
    Synthetic health data generator inspired by Synthea.
    Generates realistic patient trajectories using disease progression models.
    """
    
    def __init__(self):
        self.tokenizer = get_tokenizer()
        self.disease_names = self.tokenizer.get_disease_names()
        
        # Disease prevalence by age (estimated from real epidemiological data)
        self.age_disease_prevalence = {
            'Hypertension': {(18, 40): 0.05, (40, 60): 0.25, (60, 90): 0.60},
            'Diabetes': {(18, 40): 0.02, (40, 60): 0.12, (60, 90): 0.25},
            'Coronary Artery Disease': {(18, 40): 0.001, (40, 60): 0.05, (60, 90): 0.18},
            'Depression': {(18, 40): 0.08, (40, 60): 0.07, (60, 90): 0.05},
            'Anxiety': {(18, 40): 0.12, (40, 60): 0.09, (60, 90): 0.06},
            'Asthma': {(18, 40): 0.08, (40, 60): 0.06, (60, 90): 0.04},
            'Arthritis': {(18, 40): 0.01, (40, 60): 0.08, (60, 90): 0.30},
            'Osteoporosis': {(18, 40): 0.001, (40, 60): 0.02, (60, 90): 0.15},
            'Cancer': {(18, 40): 0.005, (40, 60): 0.02, (60, 90): 0.08},
            'Stroke': {(18, 40): 0.001, (40, 60): 0.01, (60, 90): 0.05},
        }
        
        # Disease co-occurrence patterns (higher values = more likely to co-occur)
        self.disease_correlations = {
            ('Hypertension', 'Diabetes'): 2.5,
            ('Hypertension', 'Coronary Artery Disease'): 3.0,
            ('Diabetes', 'Coronary Artery Disease'): 2.8,
            ('Depression', 'Anxiety'): 4.0,
            ('Hypertension', 'Stroke'): 2.2,
            ('Arthritis', 'Osteoporosis'): 1.8,
            ('Asthma', 'Anxiety'): 1.5,
        }
        
        # Disease progression sequences (disease A -> higher chance of disease B)
        self.disease_progressions = {
            'Hypertension': ['Coronary Artery Disease', 'Stroke', 'Heart Failure'],
            'Diabetes': ['Coronary Artery Disease', 'Kidney Disease', 'Stroke'],
            'Depression': ['Anxiety', 'Coronary Artery Disease'],
            'Asthma': ['COPD'],
            'Arthritis': ['Osteoporosis'],
        }
    
    def _get_age_prevalence(self, disease: str, age: int) -> float:
        """Get disease prevalence for a specific age"""
        if disease not in self.age_disease_prevalence:
            return 0.01  # Default low prevalence for unmapped diseases
        
        prevalence_map = self.age_disease_prevalence[disease]
        for (min_age, max_age), prevalence in prevalence_map.items():
            if min_age <= age < max_age:
                return prevalence
        return 0.005  # Very low default
    
    def _should_develop_disease(self, patient_age: int, disease: str, existing_diseases: List[str]) -> bool:
        """Determine if patient should develop a specific disease based on age and existing conditions"""
        base_prevalence = self._get_age_prevalence(disease, patient_age)
        
        # Adjust for existing diseases (co-occurrence patterns)
        adjustment_factor = 1.0
        for existing_disease in existing_diseases:
            # Check for disease correlations in both directions
            correlation_key1 = (existing_disease, disease)
            correlation_key2 = (disease, existing_disease)
            
            if correlation_key1 in self.disease_correlations:
                adjustment_factor *= self.disease_correlations[correlation_key1]
            elif correlation_key2 in self.disease_correlations:
                adjustment_factor *= self.disease_correlations[correlation_key2]
        
        adjusted_prevalence = min(base_prevalence * adjustment_factor, 0.8)  # Cap at 80%
        return random.random() < adjusted_prevalence
    
    def _generate_patient_timeline(self, patient_id: str, birth_year: int, gender: str, config: PopulationConfig) -> List[Dict]:
        """Generate complete health timeline for a single patient"""
        events = []
        current_year = max(birth_year + 18, config.start_year)  # Start at age 18 or config start year
        end_year = min(config.end_year, birth_year + 90)  # End at age 90 or config end year
        
        existing_diseases = []
        
        while current_year <= end_year:
            patient_age = current_year - birth_year
            
            # Check for new disease onset
            for disease in self.disease_names:
                if disease not in existing_diseases:
                    if self._should_develop_disease(patient_age, disease, existing_diseases):
                        # Add some randomness to the exact date within the year
                        event_date = datetime(current_year, random.randint(1, 12), random.randint(1, 28))
                        
                        events.append({
                            'patient_id': patient_id,
                            'disease_name': disease,
                            'age': patient_age + random.uniform(0, 1),  # Add fractional age
                            'event_date': event_date.strftime('%Y-%m-%d'),
                            'gender': gender
                        })
                        existing_diseases.append(disease)
            
            # Check for disease progressions
            for existing_disease in existing_diseases.copy():
                if existing_disease in self.disease_progressions:
                    for progression_disease in self.disease_progressions[existing_disease]:
                        if progression_disease not in existing_diseases and progression_disease in self.disease_names:
                            # Higher chance of progression diseases
                            if random.random() < 0.15:  # 15% chance per year
                                event_date = datetime(current_year, random.randint(1, 12), random.randint(1, 28))
                                events.append({
                                    'patient_id': patient_id,
                                    'disease_name': progression_disease,
                                    'age': patient_age + random.uniform(0, 1),
                                    'event_date': event_date.strftime('%Y-%m-%d'),
                                    'gender': gender
                                })
                                existing_diseases.append(progression_disease)
            
            current_year += 1
        
        return events
    
    def generate_population(self, config: PopulationConfig) -> pd.DataFrame:
        """
        Generate synthetic population with health trajectories
        
        Args:
            config: Population configuration parameters
            
        Returns:
            DataFrame with synthetic patient data compatible with Delphi
        """
        np.random.seed(config.seed)
        random.seed(config.seed)
        
        all_events = []
        
        print(f"Generating {config.num_patients} synthetic patients...")
        
        for i in range(config.num_patients):
            # Generate patient demographics
            patient_id = f"SYNTH_{i+1:06d}"
            
            # Random birth year based on current age distribution
            current_age = np.random.normal(50, 15)  # Normal distribution around age 50
            current_age = max(config.age_range[0], min(config.age_range[1], current_age))
            birth_year = 2024 - int(current_age)
            
            # Gender
            gender = np.random.choice(list(config.gender_distribution.keys()), 
                                    p=list(config.gender_distribution.values()))
            
            # Generate patient timeline
            patient_events = self._generate_patient_timeline(patient_id, birth_year, gender, config)
            all_events.extend(patient_events)
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{config.num_patients} patients...")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_events)
        
        if len(df) > 0:
            # Sort by patient and age
            df = df.sort_values(['patient_id', 'age']).reset_index(drop=True)
            
            print(f"Generated {len(df)} medical events for {config.num_patients} patients")
            print(f"Average events per patient: {len(df) / config.num_patients:.2f}")
            
            # Display disease distribution
            disease_counts = df['disease_name'].value_counts()
            print("\nTop 10 most common conditions:")
            print(disease_counts.head(10))
        else:
            print("Warning: No events generated!")
        
        return df
    
    def generate_population_with_conditions(self, 
                                          config: PopulationConfig,
                                          target_conditions: List[str],
                                          condition_prevalence: float = 0.3) -> pd.DataFrame:
        """
        Generate population with specific focus on certain conditions
        
        Args:
            config: Population configuration
            target_conditions: List of conditions to emphasize
            condition_prevalence: Target prevalence for specified conditions
            
        Returns:
            DataFrame with enhanced prevalence of target conditions
        """
        # Temporarily boost prevalence of target conditions
        original_prevalence = {}
        for condition in target_conditions:
            if condition in self.age_disease_prevalence:
                original_prevalence[condition] = self.age_disease_prevalence[condition].copy()
                # Boost prevalence across all age groups
                for age_range in self.age_disease_prevalence[condition]:
                    self.age_disease_prevalence[condition][age_range] *= (condition_prevalence / 0.1)
        
        # Generate population
        df = self.generate_population(config)
        
        # Restore original prevalence
        for condition, prev_map in original_prevalence.items():
            self.age_disease_prevalence[condition] = prev_map
        
        return df

def create_synthea_compatible_data(num_patients: int = 1000, 
                                 target_conditions: Optional[List[str]] = None,
                                 seed: int = 42) -> pd.DataFrame:
    """
    Convenience function to create synthetic data compatible with Delphi
    
    Args:
        num_patients: Number of synthetic patients to generate
        target_conditions: Optional list of conditions to emphasize
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame ready for use with Delphi's data pipeline
    """
    generator = SyntheaGenerator()
    config = PopulationConfig(
        num_patients=num_patients,
        seed=seed
    )
    
    if target_conditions:
        return generator.generate_population_with_conditions(config, target_conditions)
    else:
        return generator.generate_population(config)

# Example usage and presets
PRESET_CONFIGS = {
    'small': PopulationConfig(num_patients=100, seed=42),
    'medium': PopulationConfig(num_patients=1000, seed=42),
    'large': PopulationConfig(num_patients=10000, seed=42),
    'diabetes_study': PopulationConfig(num_patients=500, seed=42),
    'cardiovascular_study': PopulationConfig(num_patients=800, seed=42),
    'mental_health_study': PopulationConfig(num_patients=300, seed=42)
}

if __name__ == "__main__":
    # Test the generator
    generator = SyntheaGenerator()
    test_config = PopulationConfig(num_patients=10, seed=42)
    test_data = generator.generate_population(test_config)
    print("Test data shape:", test_data.shape)
    print("Columns:", test_data.columns.tolist())
    print("Sample data:")
    print(test_data.head())