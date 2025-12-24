"""
Quaylyn's Law - Directional Framework for Discovery Under Uncertainty

This module implements the reasoning method described in Quaylyn's Law:
movement based on directional error-reduction rather than certainty.
"""

from typing import Callable, Any, List, Tuple, Optional
from enum import Enum


class Direction(Enum):
    """Directional classification for trisection"""
    CLEARLY_WORSE = -1
    UNCERTAIN = 0
    CLEARLY_BETTER = 1


def directional_trisection(
    candidates: List[Any],
    evaluate: Callable[[Any], float],
    threshold: float = 0.3,
    max_iterations: int = 10
) -> Tuple[Any, List[Any]]:
    """
    Apply Directional Trisection to find the best candidate through elimination.
    
    Instead of claiming certainty, this function progressively eliminates what is
    clearly worse, preserving ambiguity in the uncertain middle region until
    structure emerges.
    
    Args:
        candidates: List of options to evaluate
        evaluate: Function that returns a score for each candidate (higher is better)
        threshold: Percentile threshold for trisection (default 0.3 = bottom 30% eliminated)
        max_iterations: Maximum number of elimination rounds
        
    Returns:
        Tuple of (best_candidate, elimination_history)
        
    Example:
        >>> options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> best, history = directional_trisection(options, lambda x: -abs(x - 7))
        >>> print(best)  # Should find 7 through elimination
    """
    if not candidates:
        raise ValueError("Cannot perform trisection on empty candidate list")
    
    remaining = candidates.copy()
    history = []
    
    for iteration in range(max_iterations):
        if len(remaining) == 1:
            break
            
        # Evaluate all remaining candidates
        scores = [(candidate, evaluate(candidate)) for candidate in remaining]
        scores.sort(key=lambda x: x[1])
        
        # Trisection: eliminate clearly worse (bottom threshold)
        cutoff_index = max(1, int(len(scores) * threshold))
        eliminated = [s[0] for s in scores[:cutoff_index]]
        remaining = [s[0] for s in scores[cutoff_index:]]
        
        history.append({
            'iteration': iteration + 1,
            'eliminated': eliminated,
            'remaining_count': len(remaining),
            'worst_score': scores[0][1],
            'best_score': scores[-1][1]
        })
    
    return remaining[0] if remaining else candidates[0], history


def compare_directional(
    option_a: Any,
    option_b: Any,
    evaluate: Callable[[Any], float],
    tolerance: float = 0.1
) -> Direction:
    """
    Compare two options directionally without claiming certainty.
    
    Instead of declaring which is "correct", this function identifies which
    direction reduces error, allowing for an uncertain middle ground.
    
    Args:
        option_a: First option
        option_b: Second option
        evaluate: Function to score each option
        tolerance: Range within which options are considered uncertain
        
    Returns:
        Direction indicating if A is clearly better, clearly worse, or uncertain
        
    Example:
        >>> direction = compare_directional(5, 7, lambda x: -abs(x - 6), tolerance=0.5)
        >>> # Returns CLEARLY_BETTER if A is closer to 6
    """
    score_a = evaluate(option_a)
    score_b = evaluate(option_b)
    
    difference = score_a - score_b
    
    if abs(difference) < tolerance:
        return Direction.UNCERTAIN
    elif difference > 0:
        return Direction.CLEARLY_BETTER
    else:
        return Direction.CLEARLY_WORSE


def avoid_certainty_trap(
    hypothesis: Any,
    test: Callable[[Any], bool],
    alternatives: List[Any],
    min_alternatives: int = 2
) -> dict:
    """
    Prevent premature commitment to a single hypothesis.
    
    Implements the principle: "Early commitment predicts later failure"
    
    Args:
        hypothesis: The current assumed answer
        test: Function to validate the hypothesis
        alternatives: Other possible answers
        min_alternatives: Minimum alternatives to keep alive
        
    Returns:
        Dictionary with recommendation and active options
        
    Example:
        >>> result = avoid_certainty_trap(
        ...     hypothesis="answer",
        ...     test=lambda x: x.startswith("a"),
        ...     alternatives=["alternative1", "another", "backup"]
        ... )
    """
    # Test the hypothesis
    hypothesis_valid = test(hypothesis)
    
    # Test alternatives
    valid_alternatives = [alt for alt in alternatives if test(alt)]
    
    if hypothesis_valid and len(valid_alternatives) < min_alternatives:
        return {
            'recommendation': 'DANGER: Single point of certainty detected',
            'action': 'Keep exploring alternatives',
            'active_options': [hypothesis] + alternatives,
            'certainty_level': 'TOO_HIGH'
        }
    elif hypothesis_valid and valid_alternatives:
        return {
            'recommendation': 'Multiple valid paths exist',
            'action': 'Use elimination to reduce, not assertion to select',
            'active_options': [hypothesis] + valid_alternatives,
            'certainty_level': 'HEALTHY'
        }
    elif not hypothesis_valid:
        return {
            'recommendation': 'Current hypothesis eliminated',
            'action': 'Evaluate remaining alternatives directionally',
            'active_options': valid_alternatives,
            'certainty_level': 'CORRECTING'
        }
    else:
        return {
            'recommendation': 'No clear direction yet',
            'action': 'Gather more information before elimination',
            'active_options': [hypothesis] + alternatives,
            'certainty_level': 'UNCERTAIN'
        }


def reversible_decision(
    decision_func: Callable[[], Any],
    undo_func: Optional[Callable[[], None]] = None
) -> Tuple[Any, Callable]:
    """
    Make a decision that can be reversed if it proves incorrect.
    
    Implements the principle: "Reversibility outperforms confidence"
    
    Args:
        decision_func: Function that makes the decision
        undo_func: Optional function to reverse the decision
        
    Returns:
        Tuple of (result, undo_function)
        
    Example:
        >>> def try_approach():
        ...     return "attempted solution"
        >>> def revert():
        ...     print("Reverting to previous state")
        >>> result, undo = reversible_decision(try_approach, revert)
        >>> # If result is wrong: undo()
    """
    result = decision_func()
    
    def undo():
        if undo_func:
            undo_func()
        return None
    
    return result, undo


# Example usage demonstrating the law
if __name__ == "__main__":
    print("=== Quaylyn's Law: Directional Trisection Example ===\n")
    
    # Example: Finding optimal value without claiming to know the answer
    search_space = list(range(1, 101))
    target = 73  # Hidden target
    
    def error_metric(value):
        """Lower error is better"""
        return -abs(value - target)
    
    print(f"Searching space of {len(search_space)} values...")
    print("Using directional elimination instead of certainty-based search\n")
    
    best, history = directional_trisection(
        search_space,
        error_metric,
        threshold=0.4,
        max_iterations=5
    )
    
    for step in history:
        print(f"Iteration {step['iteration']}:")
        print(f"  Eliminated {len(step['eliminated'])} clearly worse options")
        print(f"  Remaining candidates: {step['remaining_count']}")
        print(f"  Score range: [{step['worst_score']:.2f}, {step['best_score']:.2f}]")
        print()
    
    print(f"Best candidate found: {best}")
    print(f"Actual target: {target}")
    print(f"Error: {abs(best - target)}")
    print("\nNote: Discovery emerged through elimination, not certainty.")
