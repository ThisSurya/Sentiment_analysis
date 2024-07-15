from Sastrawi.Dictionary.ArrayDictionary import ArrayDictionary
from Sastrawi.StopWordRemover.StopWordRemover import StopWordRemover

def create_stop_word_factory():
    stopword = getStopWord()
    dictionary = ArrayDictionary(stopword)
    stop_word_remover = StopWordRemover(dictionary)
    
    return stop_word_remover

def getStopWord():
    return ['yang', 'untuk', 'pada', 'ke', 'para', 'namun', 'menurut', 'antara', 'dia', 'dua',
            'ia', 'seperti', 'jika', 'jika', 'sehingga', 'kembali', 'dan', 'tidak', 'ini', 'karena',
            'kepada', 'oleh', 'saat', 'harus', 'sementara', 'setelah', 'belum', 'kami', 'sekitar',
            'bagi', 'serta', 'di', 'dari', 'telah', 'sebagai', 'masih', 'hal', 'ketika', 'adalah',
            'itu', 'dalam', 'bisa', 'bahwa', 'atau', 'hanya', 'kita', 'dengan', 'akan', 'juga',
            'ada', 'mereka', 'sudah', 'saya', 'terhadap', 'secara', 'agar', 'lain', 'anda',
            'begitu', 'mengapa', 'kenapa', 'yaitu', 'yakni', 'daripada', 'itulah', 'lagi', 'maka',
            'tentang', 'demi', 'dimana', 'kemana', 'pula', 'sambil', 'sebelum', 'sesudah', 'supaya',
            'guna', 'kah', 'pun', 'sampai', 'sedangkan', 'selagi', 'sementara', 'tetapi', 'apakah',
            'kecuali', 'sebab', 'selain', 'seolah', 'seraya', 'seterusnya', 'tanpa', 'agak', 'boleh',
            'dapat', 'dsb', 'dst', 'dll', 'dahulu', 'dulunya', 'anu', 'demikian', 'tapi', 'ingin',
            'juga', 'nggak', 'mari', 'nanti', 'melainkan', 'oh', 'ok', 'seharusnya', 'sebetulnya',
            'setiap', 'setidaknya', 'sesuatu', 'pasti', 'saja', 'toh', 'ya', 'walau', 'tolong',
            'tentu', 'amat', 'apalagi', 'bagaimanapun'
            'sini', 'nomor', 'iya', 'gitu', 'cuma', 'dong', 
    'sebuah', 'sering', 'enggak', 'banyak', 'kalau', 'aku', 'nih', 'tersebut', 'dimana', 
    'begini', 'begitu', 'selalu', 'setelah', 'sebelum', 'mengenai', 'daripada', 'tersebut',
    'yakni', 'saat', 'selama', 'sepanjang', 'disebabkan', 'oleh', 'bagi', 'hanya', 'semua',
    'setiap', 'yaitu', 'yang', 'bukan', 'menjadi', 'maka', 'juga', 'kecuali', 'karena',
    'sehingga', 'jika', 'saja', 'seperti', 'dari', 'atau', 'sebab', 'namun', 'di', 'ke',
    'pada', 'tentang', 'tanpa', 'melalui', 'hingga', 'untuk', 'atas', 'dengan', 'tetapi',
    'demi', 'bahwa', 'agar', 'oleh', 'walau', 'terhadap', 'baik', 'dalam', 'termasuk', 'kita',
    'juga', 'anda', 'mereka', 'saya', 'kami', 'kau', 'ia', 'dia', 'mereka', 'hal', 'pula', 
    'pun', 'atau', 'karena', 'sehingga', 'saat', 'oleh', 'tersebut', 'gue', 'gua', 'w', 'penyempong', 'tuh', 'gwa', 'gw'
    'kakek', 'nenek', 'fun', 'math', 'bro', 'lu', 'apa', 'ada', 'moga', 'nya', 'tuh', 'alay', 'sih', 'simpson', 'malah', 'virzal', 'n'
    'sih', 'lah', 'wahyueko', 'ziroowl', 'main', 'p', 'pasar', 'kucingbahlul', 'berapa', 'kalo', 'klo', 'iblis',
    'juniartawan', 'bakal', 'jadi', 'jdi', 'bkl', 'bang', 'kayak', 'david']