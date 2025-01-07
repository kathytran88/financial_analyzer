let initialInput = document.getElementById('initial_investment');
let durationInput = document.querySelector('#duration');
let targetInput = document.getElementById('target_return');
let error = document.getElementById('errorMessage');
let button = document.getElementById('input_form_btn');

function formValidation(event) {
    event.preventDefault();

    // Check if users fill out all required inputs
    if (!initialInput || !durationInput || !targetInput) {
        error.innerHTML = 'Cannot have missing values.';
        error.style.display = "block";
    } else {
        let errors = ''; 
        let initial = parseFloat(initialInput.value);
        let duration = parseInt(durationInput.value);
        let target = parseFloat(targetInput.value);

        // Validate inputs 
        if (isNaN(initial) || initial < 50) {
            errors += 'Initial investment has to be at least 50$.<br>';
        }
        if (isNaN(duration) || duration < 1) {
            errors += 'Duration has to be at least 1 year.<br>';
        }
        if (isNaN(target) || target <= initial || target < 1) {
            errors += 'Target return has to be larger than the initial investment.<br>';
        }

        // Target too high 
        if (initial > 0 && duration > 0 && target > 0 && target > initial) {

        const maxTarget = initial * Math.pow(2.5, duration);
        const formattedMaxTarget = maxTarget.toFixed(2);
        if (target > maxTarget) {
            errors += `Target too high! For an initial investment of $${initial.toFixed(2)} and a duration of ${duration} years, the target must be ≤ $${formattedMaxTarget}.<br>`;
        }

        const ratio = target / initial;

        if (ratio >= 1.2 && ratio < 1.4 && duration < 2) {
            errors += `Your target (${target}) is ${ratio.toFixed(2)}× the initial (${initial}). ` +
                      `You must have at least 2 years for such a return.<br>`;
        }
        if (ratio >= 1.4 && ratio < 1.7 && duration < 4) {
            errors += `Your target (${target}) is ${ratio.toFixed(2)}× the initial (${initial}). ` +
                      `You must have at least 4 years for such a return.<br>`;
          }

          if (ratio >= 1.7 && ratio < 2 && duration < 5) {
            errors += `Your target (${target}) is ${ratio.toFixed(2)}× the initial (${initial}). ` +
                      `You must have at least 5 years for such a return.<br>`;
          }

 const exactMultiplier = target / initial;
  let requiredDuration = 0;
  if (exactMultiplier >= 2) {
    const n = Math.floor(exactMultiplier);

    if (n === 2) {
      requiredDuration = 5;
    } else {
      requiredDuration = 2 * n + 2; 

    }
  }

  if (duration < requiredDuration) {
    errors += "It is not feasible to achieve that high return in such a short time. " +
      "Please lower your target or increase the duration.<br>";
  }
        
        }
        // If there are any errors
        if (errors) {
            error.innerHTML = errors; 
            error.style.display = "block";
        } else {
            error.innerHTML = ''; 
            error.style.display = "none"; 
            // Allow form submission
            event.target.form.submit();
        }
    }
}

button.addEventListener('click', formValidation);
