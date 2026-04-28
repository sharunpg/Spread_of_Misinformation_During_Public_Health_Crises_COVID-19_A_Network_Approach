"""
Evaluation Framework

Purpose:
Demonstrate system correctness with curated test cases.

Approach:
- Manual test cases covering different scenarios
- Confusion matrix analysis
- Per-category accuracy
- Edge case handling

Why small curated dataset?
- B.Tech scope doesn't require big data
- Quality over quantity
- Each test case is explainable
- Demonstrates understanding of problem
"""
from typing import List, Dict, Tuple
from dataclasses import dataclass
import json
from collections import defaultdict

from classifier import MisinformationClassifier, ClassificationResult

@dataclass
class TestCase:
    """Single test case with expected outcome"""
    text: str
    expected_label: str  # CORRECT, MISINFO, UNVERIFIED
    category: str        # vaccine, treatment, transmission, etc.
    description: str     # Why this is expected

class Evaluator:
    """
    Evaluation framework for misinformation classifier.
    """
    
    def __init__(self, classifier: MisinformationClassifier = None):
        self.classifier = classifier or MisinformationClassifier()
        self.test_cases = self._create_test_suite()
    
    def _create_test_suite(self) -> List[TestCase]:
        """Create comprehensive test suite"""
        return [
            # === CORRECT INFORMATION ===
            TestCase(
                text="COVID-19 vaccines are safe and effective",
                expected_label="CORRECT",
                category="vaccine",
                description="Direct WHO/CDC guidance"
            ),
            TestCase(
                text="Wearing a mask reduces the spread of coronavirus",
                expected_label="CORRECT",
                category="prevention",
                description="Established prevention measure"
            ),
            TestCase(
                text="Social distancing helps prevent COVID transmission",
                expected_label="CORRECT",
                category="prevention",
                description="WHO recommended measure"
            ),
            TestCase(
                text="Hand washing with soap helps prevent infection",
                expected_label="CORRECT",
                category="prevention",
                description="Basic hygiene guidance"
            ),
            TestCase(
                text="COVID-19 spreads through respiratory droplets",
                expected_label="CORRECT",
                category="transmission",
                description="Scientific consensus"
            ),
            TestCase(
                text="Vaccination reduces hospitalization from COVID",
                expected_label="CORRECT",
                category="vaccine",
                description="Proven vaccine benefit"
            ),
            
            # === MISINFORMATION ===
            TestCase(
                text="5G networks spread the coronavirus",
                expected_label="MISINFO",
                category="conspiracy",
                description="Debunked conspiracy theory"
            ),
            TestCase(
                text="Drinking bleach can cure COVID-19",
                expected_label="MISINFO",
                category="false_cure",
                description="Dangerous false treatment"
            ),
            TestCase(
                text="COVID vaccines contain microchips for tracking",
                expected_label="MISINFO",
                category="conspiracy",
                description="Debunked conspiracy theory"
            ),
            TestCase(
                text="Garlic prevents coronavirus infection",
                expected_label="MISINFO",
                category="false_cure",
                description="No scientific evidence"
            ),
            TestCase(
                text="Vaccines change your DNA",
                expected_label="MISINFO",
                category="vaccine_myth",
                description="mRNA doesn't alter DNA"
            ),
            TestCase(
                text="Hot water kills coronavirus inside your body",
                expected_label="MISINFO",
                category="false_cure",
                description="WHO myth buster"
            ),
            TestCase(
                text="Masks cause oxygen deficiency",
                expected_label="MISINFO",
                category="false_claim",
                description="CDC debunked"
            ),
            TestCase(
                text="COVID-19 only affects elderly people",
                expected_label="MISINFO",
                category="false_claim",
                description="All ages can be affected"
            ),
            
            # === PARAPHRASED TESTS (harder) ===
            TestCase(
                text="The jab helps you not end up in hospital",
                expected_label="CORRECT",
                category="vaccine",
                description="Paraphrased vaccine benefit"
            ),
            TestCase(
                text="Mobile phone towers are spreading the virus",
                expected_label="MISINFO",
                category="conspiracy",
                description="Paraphrased 5G myth"
            ),
            
            # === MULTILINGUAL (tests translation) ===
            TestCase(
                text="Las vacunas COVID son seguras",  # Spanish
                expected_label="CORRECT",
                category="vaccine",
                description="Spanish - vaccines are safe"
            ),
            TestCase(
                text="Le vaccin contient des micropuces",  # French
                expected_label="MISINFO",
                category="conspiracy",
                description="French - microchip myth"
            ),
            
            # === EDGE CASES ===
            TestCase(
                text="I got my vaccine yesterday and my arm hurts",
                expected_label="CORRECT",
                category="vaccine",
                description="Normal side effect report"
            ),
            TestCase(
                text="My uncle says turmeric cured his COVID",
                expected_label="MISINFO",
                category="false_cure",
                description="Anecdotal false cure"
            ),
        ]
    
    def run_evaluation(self, verbose: bool = True) -> Dict:
        """
        Run full evaluation and return metrics.
        """
        results = []
        
        for tc in self.test_cases:
            result = self.classifier.classify(tc.text)
            
            # Normalize labels for comparison
            predicted = self._normalize_label(result.label)
            expected = self._normalize_label(tc.expected_label)
            
            correct = predicted == expected
            
            results.append({
                'text': tc.text[:50] + '...',
                'category': tc.category,
                'expected': expected,
                'predicted': predicted,
                'confidence': result.confidence,
                'correct': correct
            })
            
            if verbose:
                status = "✓" if correct else "✗"
                print(f"{status} [{tc.category}] Expected: {expected}, Got: {predicted}")
                if not correct:
                    print(f"   Text: {tc.text[:60]}...")
                    print(f"   Explanation: {result.explanation[:80]}...")
        
        # Calculate metrics
        metrics = self._calculate_metrics(results)
        
        if verbose:
            self._print_summary(metrics)
        
        return metrics
    
    def _normalize_label(self, label: str) -> str:
        """Normalize label for comparison"""
        label = label.upper()
        if 'MISINFO' in label:
            return 'MISINFO'
        elif 'CORRECT' in label:
            return 'CORRECT'
        return 'UNVERIFIED'
    
    def _calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate evaluation metrics"""
        total = len(results)
        correct = sum(1 for r in results if r['correct'])
        
        # Per-category accuracy
        by_category = defaultdict(lambda: {'correct': 0, 'total': 0})
        for r in results:
            by_category[r['category']]['total'] += 1
            if r['correct']:
                by_category[r['category']]['correct'] += 1
        
        category_accuracy = {
            cat: stats['correct'] / stats['total'] 
            for cat, stats in by_category.items()
        }
        
        # Confusion matrix
        confusion = defaultdict(lambda: defaultdict(int))
        for r in results:
            confusion[r['expected']][r['predicted']] += 1
        
        # Per-class metrics
        classes = ['CORRECT', 'MISINFO', 'UNVERIFIED']
        class_metrics = {}
        
        for cls in classes:
            tp = confusion[cls][cls]
            fp = sum(confusion[other][cls] for other in classes if other != cls)
            fn = sum(confusion[cls][other] for other in classes if other != cls)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[cls] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        return {
            'accuracy': correct / total,
            'total_tests': total,
            'correct': correct,
            'category_accuracy': category_accuracy,
            'confusion_matrix': dict(confusion),
            'class_metrics': class_metrics,
            'detailed_results': results
        }
    
    def _print_summary(self, metrics: Dict):
        """Print evaluation summary"""
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Overall Accuracy: {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total_tests']})")
        
        print("\nPer-Category Accuracy:")
        for cat, acc in sorted(metrics['category_accuracy'].items()):
            print(f"  {cat}: {acc:.1%}")
        
        print("\nPer-Class Metrics:")
        for cls, m in metrics['class_metrics'].items():
            print(f"  {cls}:")
            print(f"    Precision: {m['precision']:.2f}")
            print(f"    Recall: {m['recall']:.2f}")
            print(f"    F1: {m['f1']:.2f}")
        
        print("="*50)

def run_quick_test():
    """Quick sanity check"""
    evaluator = Evaluator()
    
    print("Running quick evaluation...\n")
    metrics = evaluator.run_evaluation(verbose=True)
    
    return metrics

if __name__ == "__main__":
    run_quick_test()