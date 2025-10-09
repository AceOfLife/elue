// Campaign Dates
document.addEventListener('DOMContentLoaded', function() {
    const startDateInput = document.getElementById('CampaignStart');
    const endDateInput = document.getElementById('CampaignEnd');
    
    // Set minimum start date to today
    const today = new Date();
    const yyyy = today.getFullYear();
    const mm = String(today.getMonth() + 1).padStart(2, '0');
    const dd = String(today.getDate()).padStart(2, '0');
    const todayFormatted = `${yyyy}-${mm}-${dd}`;
    
    startDateInput.setAttribute('min', todayFormatted);
    
    // Handle start date change
    startDateInput.addEventListener('change', function() {
        const startDate = new Date(this.value);
        const startDateTime = startDate.getTime();
        const todayStart = new Date(today);
        todayStart.setHours(0, 0, 0, 0);
        
        // Validate start date
        if (startDateTime < todayStart.getTime()) {
            // Reset if invalid
            this.value = '';
        } else {
            // Enable end date and set its minimum value
            endDateInput.disabled = false;
            const nextDay = new Date(startDateTime);
            nextDay.setDate(nextDay.getDate() + 1);
            
            const nextDayFormatted = nextDay.toISOString().split('T')[0];
            endDateInput.setAttribute('min', nextDayFormatted);
            
            // If current end date is invalid, clear it
            if (endDateInput.value) {
                const endDate = new Date(endDateInput.value);
                if (endDate <= startDate) {
                    endDateInput.value = '';
                }
            }
        }
    });
    
    // Handle end date change
    endDateInput.addEventListener('change', function() {
        if (startDateInput.value) {
            const startDate = new Date(startDateInput.value);
            const endDate = new Date(this.value);
            
            if (endDate <= startDate) {
                // Reset if invalid
                this.value = '';
            }
        }
    });
});