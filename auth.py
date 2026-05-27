from flask import request, jsonify, session
from werkzeug.security import generate_password_hash, check_password_hash
from database import get_db

def signup_user():
    data = request.get_json()

    conn = get_db()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
            (
                data['name'],
                data['email'],
                generate_password_hash(data['password'])
            )
        )
        conn.commit()
        return jsonify({"success": True, "message": "Account created successfully"})
    except Exception as e:
        return jsonify({"success": False, "message": "Email already exists"})


def login_user():
    data = request.get_json()

    email = data['email']
    password = data['password']

    # ADMIN LOGIN
    if email == "admin@gmail.com" and password == "admin123":
        session.clear()
        session['is_admin'] = True

        return jsonify({
            "success": True,
            "message": "Admin login successful",
            "redirect": "/admin"
        })

    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
    user = cursor.fetchone()

    if user and check_password_hash(user['password'], password):
        session.clear()

        session['user_id'] = user['id']
        session['user_name'] = user['name']
        session['is_admin'] = False

        return jsonify({
            "success": True,
            "message": "Login successful",
            "redirect": "/"
        })

    return jsonify({
        "success": False,
        "message": "Invalid credentials"
    })