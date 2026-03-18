d9(o5).
d5(o11).
d7(o1).
d4(o10).
d4(o2).
d12(o7).
d1(o3).
d12(o1).
d1(o13).
d3(o1).
d2(o5).
d12(o10).
d8(o4).
d3(o0).
d9(o1).

d7(X) :- not d2(X), d1(X), not d4(X).
d6(X) :- d8(X), not d7(X), not d13(X).
d5(X) :- not d1(X), not d6(X), d10(X).
d0(X) :- d8(X), not d12(X), not d11(X).
d10(X) :- not d0(X), d6(X), not d9(X).
d4(X) :- d8(X), not d6(X), not d3(X).
d10(X) :- not d3(X), d0(X), not d4(X).
d10(X) :- d8(X), not d6(X), not d9(X).
d2(X) :- not d10(X), d4(X), not d8(X).
d3(X) :- d8(X), not d11(X), not d9(X).
d13(X) :- not d3(X), d0(X), not d11(X).
d9(X) :- d8(X), not d10(X), not d12(X).
d5(X) :- d8(X), not d2(X), not d13(X).
d4(X) :- d5(X), not d6(X), not d0(X).
d11(X) :- d4(X), not d12(X), not d9(X).