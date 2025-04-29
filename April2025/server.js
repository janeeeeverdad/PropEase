// TOGGLE MENU
function toggleMenu() {
    const menu = document.getElementById('menu');
    const body = document.body;
    const uploadedBox = document.querySelector('.uploaded');
    const speechBox = document.querySelector('.speech');

    // Toggle menu open/close
    menu.classList.toggle('open');
    body.classList.toggle('menu-open');

    // Adjust box width when menu is opened or closed
    uploadedBox.classList.toggle('shrink');
    speechBox.classList.toggle('shrink');
}

// Close menu when clicking outside
document.addEventListener('click', function(event) {
    const menu = document.getElementById('menu');
    const icon = document.querySelector('.menu-icon');
    const uploadedBox = document.querySelector('.uploaded');
    const speechBox = document.querySelector('.speech');

    if (!menu.contains(event.target) && !icon.contains(event.target) && menu.classList.contains('open')) {
        menu.classList.remove('open');
        document.body.classList.remove('menu-open');

        // Ensure the boxes shrink when the menu closes
        uploadedBox.classList.remove('shrink');
        speechBox.classList.remove('shrink');
    }
});

// LOGIN FUNCTION using Fetch API to communicate with login.php
async function loginUser(event) {
    event.preventDefault(); // Prevent the default form submission

    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;

    // Send the login request to login.php
    try {
        const response = await fetch('login.php', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `email=${encodeURIComponent(email)}&password=${encodeURIComponent(password)}`
        });

        const result = await response.json(); // Parse the JSON response

        if (result.status === 'success') {
            alert(result.message); // "Login successful"
            window.location.href = 'home.html'; // Redirect to home page
        } else {
            alert(result.message); // Show error message from PHP
        }
    } catch (error) {
        console.error('Error during login:', error);
        alert('An error occurred. Please try again later.');
    }
}

// Attach the loginUser function to the login form's submit event
const loginForm = document.getElementById('login-form');
if (loginForm) {
    loginForm.addEventListener('submit', loginUser);
}

// SIGNUP FUNCTION using Fetch API to communicate with signup.php
async function signupUser(event) {
    event.preventDefault(); // Prevent the default form submission

    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;

    // Send the signup request to signup.php
    try {
        const response = await fetch('signup.php', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `email=${encodeURIComponent(email)}&password=${encodeURIComponent(password)}`
        });

        const result = await response.json(); // Parse the JSON response

        if (result.status === 'success') {
            alert(result.message); // "Signup successful"
            window.location.href = 'main.html'; // Redirect to login page
        } else {
            alert(result.message); // Show error message from PHP
        }
    } catch (error) {
        console.error('Error during signup:', error);
        alert('An error occurred. Please try again later.');
    }
}

// Attach the signupUser function to the signup form's submit event
const signupForm = document.getElementById('signup-form');
if (signupForm) {
    signupForm.addEventListener('submit', signupUser);
}

// Fetch user details and display on the account page
window.addEventListener('DOMContentLoaded', async () => {
    if (window.location.pathname.endsWith('account.html')) {
        try {
            const response = await fetch('get_user_details.php');
            const result = await response.json();

            if (result.status === 'success') {
                // Populate email and mask password with asterisks
                document.getElementById('email-display').textContent = result.email;
                document.getElementById('password-display').textContent = '*'.repeat(result.password.length);
            } else {
                // Display error and redirect to main page
                alert(result.message);
                window.location.href = 'main.html';
            }
        } catch (error) {
            // Handle network errors or unexpected issues
            alert('An error occurred while fetching user details. Please try again later.');
            console.error('Error:', error);
        }
    }
});

// Delete Account
function confirmDelete() {
    const confirmation = confirm("Are you sure you want to delete your account? This action cannot be undone.");
    if (confirmation) {
        deleteAccount();
    }
}

async function deleteAccount() {
    try {
        const response = await fetch('delete_account.php', {
            method: 'POST',
        });
        const result = await response.json();

        if (result.status === 'success') {
            alert('Your account has been deleted.');
            window.location.href = 'main.html'; // Redirect to main page after deletion
        } else {
            alert(result.message); // Show error message if any
        }
    } catch (error) {
        console.error('Error during account deletion:', error);
        alert('An error occurred while deleting your account.');
    }
}