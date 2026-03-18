d2(o13).
d6(o1).
d8(o17).
d2(o10).
d8(o2).
d0(o16).
d9(o8).
d11(o13).
d5(o17).
d11(o4).

d3(X) :- d8(X), not d4(X).
d10(X) :- d9(X), not d6(X).
d8(X) :- not d9(X), d11(X).
d1(X) :- not d3(X), d10(X).
d8(X) :- d1(X), not d11(X).
d13(X) :- not d6(X), d10(X).
d12(X) :- not d3(X), d10(X).
d6(X) :- not d0(X), d11(X).
d10(X) :- not d0(X), d3(X).
d4(X) :- not d6(X), d11(X).
d8(X) :- d0(X), not d9(X).
d3(X) :- not d6(X), d10(X).
d2(X) :- d0(X), not d11(X).
d8(X) :- not d1(X), d0(X).
d9(X) :- not d0(X), d4(X).