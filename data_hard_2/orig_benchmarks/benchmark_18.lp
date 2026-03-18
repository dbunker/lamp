d0(o15).
d15(o1).
d16(o11).
d11(o1).
d17(o0).
d2(o8).
d10(o11).
d1(o12).
d10(o8).
d9(o11).

d10(X) :- not d0(X), d7(X).
d14(X) :- d12(X), not d6(X).
d7(X) :- not d0(X), d13(X).
d8(X) :- not d0(X), d12(X).
d5(X) :- not d8(X), d13(X).
d4(X) :- not d17(X), d9(X).
d5(X) :- not d10(X), d6(X).
d15(X) :- d4(X), not d10(X).
d0(X) :- not d1(X), d17(X).
d14(X) :- d1(X), not d6(X).
d10(X) :- d17(X), not d16(X).
d6(X) :- d2(X), not d3(X).
d1(X) :- d4(X), not d15(X).
d1(X) :- d3(X), not d15(X).
d5(X) :- d16(X), not d2(X).
d13(X) :- not d11(X), d15(X).
d12(X) :- d7(X), not d15(X).
d5(X) :- not d4(X), d15(X).
d3(X) :- d12(X), not d9(X).
d1(X) :- d7(X), not d11(X).