d12(o11).
d12(o2).
d10(o11).
d9(o2).
d5(o4).
d12(o8).
d11(o1).
d7(o13).
d6(o10).
d9(o6).
d11(o10).
d1(o9).
d2(o5).
d8(o4).
d13(o6).

d2(X) :- d3(X), not d8(X).
d5(X) :- not d10(X), d4(X).
d4(X) :- d2(X), not d10(X).
d9(X) :- not d0(X), d1(X).
d10(X) :- d8(X), not d2(X).
d2(X) :- not d1(X), d6(X).
d0(X) :- d4(X), not d8(X).
d3(X) :- d2(X), not d13(X).
d8(X) :- d9(X), not d2(X).
d12(X) :- not d2(X), d3(X).
d10(X) :- d8(X), not d4(X).
d9(X) :- d7(X), not d3(X).
d12(X) :- d7(X), not d0(X).
d12(X) :- not d3(X), d13(X).
d4(X) :- not d2(X), d11(X).