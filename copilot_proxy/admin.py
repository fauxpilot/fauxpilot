from firebase_admin import auth, credentials, initialize_app

admin_credential = credentials.Certificate('../serviceAccountKey.json')
initialize_app(admin_credential)

user = auth.get_user_by_email('sagimedina@gmail.com')
auth.set_custom_user_claims(user.uid, {'allowed': True})
