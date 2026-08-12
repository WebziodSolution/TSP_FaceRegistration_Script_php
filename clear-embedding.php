<?php
/**
 * Standalone Face Recognition Clear Embedding API
 * Clears the registered face embedding for a specific user ID.
 */

// CORS & Referrer Policy headers
$origin = isset($_SERVER['HTTP_ORIGIN']) ? $_SERVER['HTTP_ORIGIN'] : '';
// if (empty($origin) && isset($_SERVER['HTTP_REFERER'])) {
//     $parsed_referer = parse_url($_SERVER['HTTP_REFERER']);
//     if (isset($parsed_referer['scheme']) && isset($parsed_referer['host'])) {
//         $origin = $parsed_referer['scheme'] . '://' . $parsed_referer['host'];
//         if (isset($parsed_referer['port'])) {
//             $origin .= ':' . $parsed_referer['port'];
//         }
//     }
// }
// if (empty($origin) || $origin === '*') {
//     $origin = 'https://calcsalary.ematrixinfotech.com';
// }

if (!empty($origin)) {
    header("Access-Control-Allow-Origin: " . $origin);
} else {
    header("Access-Control-Allow-Origin: *");
}
header("Access-Control-Allow-Credentials: true");
header("Access-Control-Allow-Methods: POST, DELETE, OPTIONS");
header("Access-Control-Allow-Headers: Content-Type, Authorization, X-Requested-With");
header("Referrer-Policy: strict-origin-when-cross-origin");

if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(200);
    exit;
}

// --- Configuration ---
$env = 'demo'; // Options: local, dev, prod
$db_urls = [
    'local' => 'mysql+mysqlconnector://root:@localhost/calcsalary',
    'prod'  => 'mysql+mysqlconnector://admin:01eMatrix007!@69.57.172.154:3306/ematrix_calcsalary',
    'demo'  => 'mysql+mysqlconnector://admin:01eMatrix007!@69.57.172.154:3306/demo_calcsalary',
];
define('FACE_MATCH_THRESHOLD', 0.45);

// IP restriction: apply only in production
$client_ip = isset($_SERVER['REMOTE_ADDR']) ? $_SERVER['REMOTE_ADDR'] : '';
if (!empty($_SERVER['HTTP_X_FORWARDED_FOR'])) {
    $ips = explode(',', $_SERVER['HTTP_X_FORWARDED_FOR']);
    $client_ip = trim($ips[0]);
} elseif (!empty($_SERVER['HTTP_CLIENT_IP'])) {
    $client_ip = $_SERVER['HTTP_CLIENT_IP'];
}

$is_allowed = true;
if ($env === 'prod') {
    $is_allowed = false;
    if (strpos($client_ip, '192.168.') === 0 || $client_ip === '127.0.0.1' || $client_ip === '150.129.166.66' || $client_ip === '::1') {
        $is_allowed = true;
    }
}

if (!$is_allowed) {
    http_response_code(403);
    header('Content-Type: application/json');
    echo json_encode(["detail" => "Forbidden: Access denied from IP " . $client_ip]);
    exit;
}

$method = $_SERVER['REQUEST_METHOD'];
if ($method !== 'POST' && $method !== 'DELETE') {
    http_response_code(405);
    header('Content-Type: application/json');
    echo json_encode(["detail" => "Method Not Allowed. Use DELETE or POST."]);
    exit;
}

// --- DB URL Parser ---
function parse_db_url($url) {
    $cleaned_url = str_replace('mysql+mysqlconnector://', 'mysql://', $url);
    $parsed = parse_url($cleaned_url);
    if (!$parsed) {
        throw new Exception("Failed to parse database URL.");
    }
    
    $host = isset($parsed['host']) ? $parsed['host'] : 'localhost';
    $port = isset($parsed['port']) ? $parsed['port'] : '3306';
    $user = isset($parsed['user']) ? urldecode($parsed['user']) : 'root';
    $pass = isset($parsed['pass']) ? urldecode($parsed['pass']) : '';
    $dbname = isset($parsed['path']) ? ltrim($parsed['path'], '/') : '';
    
    return [
        'host' => $host,
        'port' => $port,
        'user' => $user,
        'pass' => $pass,
        'dbname' => $dbname
    ];
}

// --- Extract User ID ---
$userId = null;
$uri = parse_url($_SERVER['REQUEST_URI'], PHP_URL_PATH);

if (preg_match('/\/clear-embedding(?:\.php)?\/(\d+)\/?$/', $uri, $matches)) {
    $userId = intval($matches[1]);
}

if ($userId === null) {
    if (isset($_GET['user_id'])) {
        $userId = intval($_GET['user_id']);
    } elseif (isset($_GET['id'])) {
        $userId = intval($_GET['id']);
    }
}

if ($userId === null) {
    $data = [];
    $json = file_get_contents('php://input');
    if ($json) {
        $decoded = json_decode($json, true);
        if (is_array($decoded)) {
            $data = $decoded;
        }
    }
    if (isset($data['user_id'])) {
        $userId = intval($data['user_id']);
    } elseif (isset($data['id'])) {
        $userId = intval($data['id']);
    }
}

header('Content-Type: application/json');

if ($userId === null) {
    http_response_code(400);
    echo json_encode(["detail" => "user_id is a required parameter."]);
    exit;
}

// --- Database Connection ---
$db_url = isset($db_urls[$env]) ? $db_urls[$env] : null;
if (!$db_url) {
    http_response_code(500);
    echo json_encode(["detail" => "Database URL configuration for environment '$env' not found."]);
    exit;
}

try {
    $db_config = parse_db_url($db_url);
    $dsn = "mysql:host={$db_config['host']};port={$db_config['port']};dbname={$db_config['dbname']};charset=utf8mb4";
    $options = [
        PDO::ATTR_ERRMODE            => PDO::ERRMODE_EXCEPTION,
        PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
        PDO::ATTR_EMULATE_PREPARES   => false,
    ];
    $db = new PDO($dsn, $db_config['user'], $db_config['pass'], $options);
} catch (Exception $e) {
    http_response_code(500);
    echo json_encode(["detail" => "Database connection failed: " . $e->getMessage()]);
    exit;
}

// --- Clear Embedding Business Logic ---
try {
    $stmt = $db->prepare("SELECT id, embedding FROM company_employees WHERE id = :id");
    $stmt->execute(['id' => $userId]);
    $user = $stmt->fetch();

    if (!$user) {
        http_response_code(404);
        echo json_encode(["detail" => "User not found."]);
        exit;
    }

    if ($user['embedding'] !== null) {
        $db->beginTransaction();
        $update_stmt = $db->prepare("UPDATE company_employees SET embedding = NULL WHERE id = :id");
        $update_stmt->execute(['id' => $userId]);
        $db->commit();
    }

    echo json_encode([
        "message" => "User face recognition data cleared successfully",
        "status" => "success"
    ]);

} catch (Exception $e) {
    if ($db->inTransaction()) {
        $db->rollBack();
    }
    http_response_code(500);
    echo json_encode(["detail" => "Failed to clear embedding: " . $e->getMessage()]);
}
