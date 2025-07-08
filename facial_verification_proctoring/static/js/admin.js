document.addEventListener('DOMContentLoaded', function() {
    const loginForm = document.getElementById('login-form');
    const registerForm = document.getElementById('register-form');
    const errorMessage = document.getElementById('error-message');

    if (loginForm) {
        loginForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            const username = document.getElementById('username').value;
            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;

            try {
                const response = await fetch('/api/login', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ username, email, password })
                });

                const data = await response.json();

                if (response.ok) {
                    window.location.href = data.redirect;
                } else {
                    errorMessage.textContent = data.error;
                    errorMessage.style.display = 'block';
                }
            } catch (error) {
                errorMessage.textContent = 'An error occurred. Please try again.';
                errorMessage.style.display = 'block';
            }
        });
    }

    if (registerForm) {
        registerForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            const name = document.getElementById('name').value;
            const username = document.getElementById('username').value;
            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;
            const confirmPassword = document.getElementById('confirm-password').value;

            if (password !== confirmPassword) {
                errorMessage.textContent = 'Passwords do not match';
                errorMessage.style.display = 'block';
                return;
            }

            try {
                const response = await fetch('/api/admin/register', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ name, username, email, password })
                });

                const data = await response.json();

                if (response.ok) {
                    window.location.href = '/admin-login';
                } else {
                    errorMessage.textContent = data.error;
                    errorMessage.style.display = 'block';
                }
            } catch (error) {
                errorMessage.textContent = 'An error occurred. Please try again.';
                errorMessage.style.display = 'block';
            }
        });
    }
}); 