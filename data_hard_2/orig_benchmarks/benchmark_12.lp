d5(o0).
d12(o9).
d16(o7).
d7(o8).
d9(o10).
d6(o9).
d12(o0).
d6(o7).
d3(o4).
d10(o13).

d3(X) :- not d4(X), d12(X).
d17(X) :- not d3(X), d6(X).
d2(X) :- not d0(X), d3(X).
d9(X) :- not d1(X), d8(X).
d4(X) :- not d11(X), d5(X).
d14(X) :- d2(X), not d3(X).
d2(X) :- d8(X), not d11(X).
d0(X) :- not d13(X), d6(X).
d8(X) :- not d10(X), d15(X).
d14(X) :- not d4(X), d3(X).
d15(X) :- not d7(X), d3(X).
d16(X) :- not d10(X), d14(X).
d9(X) :- not d7(X), d6(X).
d4(X) :- not d7(X), d11(X).
d2(X) :- not d14(X), d3(X).
d13(X) :- d7(X), not d15(X).
d8(X) :- not d1(X), d13(X).
d11(X) :- not d8(X), d3(X).
d8(X) :- not d17(X), d16(X).
d1(X) :- d12(X), not d3(X).