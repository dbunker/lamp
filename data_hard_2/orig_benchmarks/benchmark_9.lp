d8(o7).
d5(o3).
d3(o10).
d1(o4).
d13(o11).
d0(o13).
d8(o13).
d3(o12).
d11(o13).
d13(o13).
d10(o6).
d12(o13).
d10(o8).
d9(o8).
d1(o8).

d13(X) :- d6(X), not d2(X).
d3(X) :- not d11(X), d5(X).
d0(X) :- not d8(X), d7(X).
d4(X) :- not d8(X), d1(X).
d8(X) :- d10(X), not d2(X).
d8(X) :- d10(X), not d2(X).
d11(X) :- not d7(X), d5(X).
d6(X) :- not d0(X), d7(X).
d0(X) :- not d4(X), d12(X).
d9(X) :- d4(X), not d2(X).
d8(X) :- not d0(X), d5(X).
d7(X) :- d9(X), not d2(X).
d3(X) :- d4(X), not d12(X).
d13(X) :- d2(X), not d7(X).
d2(X) :- not d6(X), d3(X).
d0(X) :- not d7(X), d13(X).
d1(X) :- not d4(X), d7(X).
d1(X) :- d4(X), not d12(X).
d9(X) :- not d10(X), d7(X).
d12(X) :- d0(X), not d5(X).