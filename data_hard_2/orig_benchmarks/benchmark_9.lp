d10(o14).
d9(o17).
d8(o16).
d3(o4).
d11(o16).
d5(o6).
d8(o0).
d1(o0).
d8(o7).
d6(o0).
d7(o9).
d13(o4).
d2(o5).
d0(o16).
d12(o0).

d1(X) :- not d7(X), d12(X).
d7(X) :- d0(X), not d11(X).
d9(X) :- not d3(X), d11(X).
d8(X) :- d1(X), not d11(X).
d12(X) :- d6(X), not d5(X).
d8(X) :- d9(X), not d10(X).
d2(X) :- not d10(X), d4(X).
d5(X) :- d2(X), not d8(X).
d3(X) :- d5(X), not d8(X).
d0(X) :- not d8(X), d11(X).
d12(X) :- d0(X), not d4(X).
d0(X) :- d1(X), not d12(X).
d7(X) :- not d2(X), d13(X).
d4(X) :- d3(X), not d13(X).
d10(X) :- not d0(X), d3(X).