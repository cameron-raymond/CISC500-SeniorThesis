module List

sig Node {
    next : option Node
} 

sig List {
    first : Node
}

fact NodeInOneList {
    all n : Node | one l : List | n in (l.first).*next
}

fact NoCycle {
    all n : Node | n ! in n.^next
}

fun Show() {}

run Show for 4