import { Pool } from 'pg';

const pool = new Pool({
  host: 'localhost',
  user: 'user',
  password: 'password',
  database: 'mydatabase'
});

async function getCustomerInfo(email) {
  const customer = await getCustomer(email);
  const purchases = await getCustomerPurchases(customer);

  return {
    customer: customer,
    purchases: purchases
  }
}

async function getCustomer(email) {
  const query = "SELECT * FROM customers WHERE email = $1"
  return (await pool.query(query, [email])).rows[0];
}

async function getCustomerPurchases(customer) {
  const query = "SELECT * FROM purchases WHERE customer_id = $1";
  return (await pool.query(query, [customer.id])).rows;
}

export { getCustomerInfo }