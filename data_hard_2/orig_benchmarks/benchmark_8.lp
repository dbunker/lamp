d12(o9).
d2(o11).
d3(o4).
d0(o15).
d9(o12).
d5(o4).
d9(o6).
d3(o17).
d12(o0).
d6(o11).

d5(X) :- not d1(X), d6(X), not d12(X).
d9(X) :- not d1(X), d10(X), not d3(X).
d4(X) :- d5(X), not d2(X), not d13(X).
d7(X) :- d0(X), not d13(X), not d8(X).
d10(X) :- d0(X), not d11(X), not d2(X).
d11(X) :- d7(X), not d1(X), not d13(X).
d7(X) :- not d3(X), not d6(X), d12(X).
d6(X) :- d7(X), not d4(X), not d11(X).
d1(X) :- d0(X), not d6(X), not d4(X).
d9(X) :- d5(X), not d1(X), not d8(X).
d8(X) :- not d6(X), not d9(X), d13(X).
d12(X) :- d0(X), not d6(X), not d9(X).
d4(X) :- d0(X), not d2(X), not d8(X).
d13(X) :- not d6(X), d1(X), not d12(X).
d2(X) :- d9(X), not d12(X), not d0(X).