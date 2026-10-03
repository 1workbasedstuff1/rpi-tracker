use std::net::UdpSocket;

fn main() {
    let socket = UdpSocket::bind("0.0.0.0:5000").unwrap();
    let mut buf = [0u8; 8];

    loop {
        println!("in loop");
        let (_, from) = socket.recv_from(&mut buf).unwrap();
        println!("{} from {}", u64::from_be_bytes(buf), from);
    }
}
