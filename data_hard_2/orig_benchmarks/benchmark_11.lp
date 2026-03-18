d0(o15).
d9(o15).
d0(o17).
d8(o7).
d3(o10).
d4(o13).
d6(o16).
d2(o10).
d1(o14).
d3(o7).
d1(o10).
d1(o9).
d13(o16).
d12(o1).
d11(o6).

d8(X) :- not d1(X), not d12(X), d3(X).
d12(X) :- not d13(X), d7(X), not d10(X).
d13(X) :- d7(X), not d11(X), not d9(X).
d7(X) :- d2(X), not d6(X), not d3(X).
d9(X) :- not d7(X), d0(X), not d10(X).
d4(X) :- d2(X), not d13(X), not d5(X).
d6(X) :- not d13(X), d1(X), not d12(X).
d2(X) :- not d7(X), d8(X), not d9(X).
d11(X) :- d4(X), not d5(X), not d12(X).
d3(X) :- not d0(X), d1(X), not d6(X).
d7(X) :- not d1(X), not d12(X), d10(X).
d9(X) :- not d7(X), d8(X), not d1(X).
d4(X) :- not d12(X), d9(X), not d2(X).
d3(X) :- not d10(X), d7(X), not d5(X).
d3(X) :- not d4(X), not d12(X), d9(X).
d12(X) :- d9(X), not d5(X), not d2(X).
d8(X) :- not d3(X), not d6(X), d13(X).
d6(X) :- d8(X), not d3(X), not d10(X).
d5(X) :- not d0(X), d12(X), not d3(X).
d10(X) :- not d1(X), not d0(X), d9(X).