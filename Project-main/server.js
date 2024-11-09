// TOGGLE MENU
function toggleMenu() {
    const menu = document.getElementById('menu');
    menu.style.left = menu.style.left === '0px' ? '-300px' : '0px';
}

// Close menu when clicking outside of it
document.addEventListener('click', function(event) {
    const menu = document.getElementById('menu');
    const icon = document.querySelector('.menu-icon');
    if (!menu.contains(event.target) && !icon.contains(event.target)) {
        menu.style.left = '-300px'; // Hide the menu
    }
});

// LOGIN FUNCTION
function loginUser(event) {
    event.preventDefault();

    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;

    if (email === '' || password === '') {
        alert('Please enter both email and password');
        return;
    }

    sessionStorage.setItem('userEmail', email);
    sessionStorage.setItem('userPassword', password);

    window.location.href = 'home.html';
}

// ACCOUNT PAGE LOAD FUNCTION
window.addEventListener('DOMContentLoaded', () => {
    if (window.location.pathname.endsWith('account.html')) {
        const email = sessionStorage.getItem('userEmail');
        const password = sessionStorage.getItem('userPassword');

        if (email && password) {
            document.getElementById('email-display').textContent = email;
            document.getElementById('password-display').textContent = '*'.repeat(password.length);
        } else {
            window.location.href = 'login.html';
        }
    }
});

// DELETE ACCOUNT FUNCTION
function confirmDelete() {
    const confirmation = confirm("Are you sure you want to delete your account? This action cannot be undone.");
    if (confirmation) {
        deleteAccount();
    }
}

function deleteAccount() {
    sessionStorage.removeItem('userEmail');
    sessionStorage.removeItem('userPassword');
    alert("Your account has been deleted.");
    window.location.href = "main.html";
}
